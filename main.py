from sqlalchemy.engine import URL

db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="similarity_search_service_db"
)

from typing import List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Create the base class for the table definition
class Base(DeclarativeBase):
    __abstract__ = True


# Create the table definition
class Images(Base):
    __tablename__ = "images"
    VECTOR_LENGTH = 512

    # primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # image path - we will use it to store the path to the image file, after similarity search we can use it to retrieve the image and display it
    image_path: Mapped[str] = mapped_column(String(256))
    # image embedding - we will store the image embedding in this column, the image embedding is a list of 512 floats this is the output of the sentence transformer model
    image_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))

from sqlalchemy import create_engine

engine = create_engine(db_url)

Base.metadata.create_all(engine)


from sqlalchemy.orm import Session
from sqlalchemy import Engine, select
import numpy as np

# reusable function to insert data into the table
def insert_image(engine: Engine, image_path: str, image_embedding: list[float]):
    with Session(engine) as session:
        # create the image object
        image = Images(
            image_path=image_path, image_embedding=image_embedding)
        # add the image object to the session
        session.add(image)
        # commit the transaction
        session.commit()

# insert some data into the table
N = 100
for i in range(N):
    image_path = f"image_{i}.jpg"
    image_embedding = np.random.rand(512).tolist()
    insert_image(engine, image_path, image_embedding)

# select first image from the table
with Session(engine) as session:
    image = session.query(Images).first()


# Note: pgvector exposes *distances*, not similarities (lower = more similar).
# To rank, sort ascending by `cosine_distance`. To filter by a similarity threshold,
# use `1 - cosine_distance > threshold`.

# order the K images by ascending cosine distance to the first image (smallest distance = most similar)
def find_k_images(engine: Engine, k: int, original_image: Images) -> list[Images]:
    with Session(engine) as session:
        # execution_options={"prebuffer_rows": True} is used to prebuffer the rows, this is useful when we want to fetch the rows in chunks and return them after session is closed
        result = session.execute(
            select(Images)
            .order_by(Images.image_embedding.cosine_distance(original_image.image_embedding))
            .limit(k),
            execution_options={"prebuffer_rows": True}
        ).scalars().all()
        return result

# find the 10 most similar images to the first image
k = 10
similar_images = find_k_images(engine, k, image)


# find images with similarity to the original above the threshold (similarity = 1 - cosine_distance)
def find_images_with_similarity_score_greater_than(engine: Engine, similarity_score: float, original_image: Images) -> list[Images]:
    with Session(engine) as session:
        result = session.execute(
            select(Images)
            .filter((1 - Images.image_embedding.cosine_distance(original_image.image_embedding)) > similarity_score),
            execution_options={"prebuffer_rows": True}
        ).scalars().all()
        return result

from datasets import load_dataset

dataset = load_dataset("FronkonGames/steam-games-dataset")

# get columns names and types
columns = dataset["train"].features
print(columns)

columns_to_keep = ["name", "windows", "linux", "mac", "detailed_description", "supported_languages", "price"]

N = 40000 # you can adjust this number
dataset = dataset["train"].select_columns(columns_to_keep).select(range(N))

from sqlalchemy import Integer, Float, Boolean, Text


class Games(Base):
    __tablename__ = "games"
    __table_args__ = {'extend_existing': True}

    # the vector size produced by the model taken from documentation https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
    VECTOR_LENGTH =  512 # check the model output dimensionality

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    # `Text` is an unbounded string in Postgres — game descriptions can run a few KB
    description: Mapped[str] = mapped_column(Text)
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH)) # fill it in proper way


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


from sentence_transformers import SentenceTransformer


checkpoint = "distiluse-base-multilingual-cased-v2"
model = SentenceTransformer(checkpoint)


def generate_embeddings(text: str) -> list[float]:
    return model.encode(text)


from tqdm import tqdm


def insert_games(engine, dataset):
    with tqdm(total=len(dataset)) as pbar:
       for i, game in enumerate(dataset):
           game_description = game["detailed_description"] or ""
           game_embedding = generate_embeddings(game_description)
           name, windows, linux, mac, price = game["name"], game["windows"], game["linux"], game["mac"], game["price"]
           # keep rows with the required fields; windows/linux/mac are platform flags (True/False),
           # not missing-data signals — don't gate the insert on them or you discard most of the dataset
           if name and game_description and price is not None:
               game = Games(
                   name=game["name"],
                   description=game_description,
                   windows=game["windows"],
                   linux=game["linux"],
                   mac=game["mac"],
                   price=game["price"],
                   game_description_embedding=game_embedding
               )
               with Session(engine) as session:
                   session.add(game)
                   session.commit()
           pbar.update(1)

insert_games(engine, dataset)


def find_game(
        engine: Engine,
        game_description: str,
        windows: Optional[bool] = None,
        linux: Optional[bool] = None,
        mac: Optional[bool] = None,
        price: Optional[int] = None
):
    with Session(engine) as session:
        game_embedding = generate_embeddings(game_description)  # generate game embedding

        query = (
            select(Games)
            .order_by(Games.game_description_embedding.cosine_distance(game_embedding))
        )

        if price:
            query = query.filter(Games.price <= price)
        if windows:
            query = query.filter(Games.windows == True)
        if linux:
            query = query.filter(Games.linux == True)
        if mac:
            query = query.filter(Games.mac == True)

        result = session.execute(query, execution_options={"prebuffer_rows": True})
        game = result.scalars().first()

        return game

game = find_game(engine, "This is a game about a hero who saves the world", price=10)
print(f"Game: {game.name}")
print(f"Description: {game.description}")

game = find_game(engine, game_description="Home decorating", price=20)
print(f"Game: {game.name}")
print(f"Description: {game.description}")


game = find_game(engine, game_description="Home decorating", mac=True, price=5)
print(f"Game: {game.name}")
print(f"Description: {game.description}")
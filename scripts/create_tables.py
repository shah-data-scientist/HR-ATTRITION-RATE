from database.database import Base, engine

if __name__ == "__main__":
    print("Creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    print("Done.")

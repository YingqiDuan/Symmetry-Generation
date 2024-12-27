from generator import Generator
import time


def main():
    g = Generator()
    g.load_model()
    g.get_embeddings()


if __name__ == "__main__":
    start = time.time()
    main()

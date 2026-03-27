import os
from services.regime_transitions import RegimeTransitionService


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    service = RegimeTransitionService(base_dir)
    service.run()


if __name__ == "__main__":
    main()
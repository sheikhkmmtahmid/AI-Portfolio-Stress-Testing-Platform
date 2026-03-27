import os

# Set environment variables BEFORE importing sklearn-related modules
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"   # change to your preferred CPU count if needed

from services.regime_detection import RegimeDetectionService


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    service = RegimeDetectionService(base_dir=base_dir)
    service.run()


if __name__ == "__main__":
    main()
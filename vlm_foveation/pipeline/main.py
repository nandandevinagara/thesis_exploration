"""Entry point for Vision–Language Model foveation experiments."""

from helper.argument_reader import get_args
from helper.misc_utils import create_log_and_csv_files


def main():
    """Parse CLI arguments and orchestrate the foveation pipeline."""
    args = get_args()
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Initialize logging artifacts for the selected dataset/log directory pair.
    create_log_and_csv_files(log_dir=args.log_dir, dataset_name=args.dataset)

    # TODO: Load the requested dataset split and metadata into memory-efficient buffers.
    # TODO: Run the foveation/sampling procedure before image-language encoding.
    # TODO: Invoke the configured VLM for reasoning or answer generation on foveated inputs.
    # TODO: Evaluate accuracy, efficiency, and qualitative signals; persist metrics to log_dir.


if __name__ == "__main__":
    main()

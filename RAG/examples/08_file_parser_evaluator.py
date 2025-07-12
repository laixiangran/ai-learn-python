import glob
import os
import time
from RAG.libs.file_parser import FileParser


def mineru_parser():
    file_parser = FileParser(provider="MinerUParser")
    all_images = glob.glob(os.path.join("RAG/datas/omnidocbench_pdf", "*"))
    output_dir = "RAG/outputs/MinerU_parser_data/eval"
    logs = []
    total_time = 0
    for image in all_images:
        start_time = time.time()
        file_parser.parse_doc(path_list=[image], output_dir=output_dir)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Processed {image} in {elapsed_time:.2f} seconds")
        logs.append(f"Processed {image} in {elapsed_time:.2f} seconds")
        with open(os.path.join(output_dir, "mineru_parser_log.txt"), "w") as log_file:
            for log in logs:
                log_file.write(log + "\n")
    print(f"Total time for parsing all images: {total_time:.2f} seconds")
    logs.append(f"Total time for parsing all images: {total_time:.2f} seconds")
    with open(os.path.join(output_dir, "mineru_parser_log.txt"), "w") as log_file:
        for log in logs:
            log_file.write(log + "\n")


if __name__ == "__main__":
    mineru_parser()

from RAG.libs.file_parser import FileParser


def PPStructureV3_parser_test():
    file_parser = FileParser(provider="PPStructureV3")
    path_list = [
        "RAG/datas/omnidocbench_pdf/docstructbench_00039896.1983.10545823.pdf_1.jpg",
        "RAG/datas/2024少儿编程教育行业发展趋势报告_simple.pdf",
    ]
    output_dir = "RAG/outputs/PPStructureV3_parser_data/test/"
    file_parser.parse_doc(path_list=path_list, output_dir=output_dir)


def MinerU_parser_test():
    file_parser = FileParser(provider="MinerU")
    path_list = [
        "RAG/datas/omnidocbench_pdf/docstructbench_00039896.1983.10545823.pdf_1.jpg",
        "RAG/datas/2024少儿编程教育行业发展趋势报告_simple.pdf",
    ]
    output_dir = "RAG/outputs/MinerU_parser_data/test/"
    file_parser.parse_doc(path_list=path_list, output_dir=output_dir)


def Docling_parser_test():
    file_parser = FileParser(provider="Docling")
    path_list = [
        "RAG/datas/omnidocbench_pdf/docstructbench_00039896.1983.10545823.pdf_1.jpg",
        "RAG/datas/2024少儿编程教育行业发展趋势报告_simple.pdf",
    ]
    output_dir = "RAG/outputs/Docling_parser_data/test/"
    file_parser.parse_doc(path_list=path_list, output_dir=output_dir)


if __name__ == "__main__":
    MinerU_parser_test()
    # PPStructureV3_parser_test()
    # Docling_parser_test()

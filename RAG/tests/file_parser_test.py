from RAG.libs.file_parser import FileParser


def mineru_parser_test():
    #  创建一个解析器对象
    file_parser = FileParser(provider="minerUParser")

    # pdf 文件解析
    file_parser.convert_pdf(file_path="RAG/datas/2024少儿编程教育行业发展趋势报告.pdf")

    # 微软办公文件解析
    file_parser.convert_ms_office(
        file_path="RAG/datas/2024少儿编程教育行业发展趋势报告.docx"
    )

    # 单个图片文件解析
    file_parser.convert_image(file_path="RAG/datas/images/跨学科融合的少儿编程教育.jpg")

    # 多个图片文件解析
    file_parser.convert_image(file_path="RAG/datas/images/")


if __name__ == "__main__":
    mineru_parser_test()

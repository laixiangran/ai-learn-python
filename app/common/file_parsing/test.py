from parsing_with_MinerU import MinerUParser

if __name__ == "__main__":
    #  创建一个解析器对象
    parserWithMinerU = MinerUParser()

    # pdf 文件解析
    parserWithMinerU.convert_pdf(
        "data/2024少儿编程教育行业发展趋势报告.pdf")

    # 微软办公文件解析
    parserWithMinerU.convert_ms_office(
        "data/2024少儿编程教育行业发展趋势报告.docx")

    # 单个图片文件解析
    parserWithMinerU.convert_image(
        "data/跨学科融合的少儿编程教育.jpg")

    # 多个图片文件解析
    parserWithMinerU.is_image_single = False
    parserWithMinerU.convert_image(
        "data/images/")

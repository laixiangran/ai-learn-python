import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_office


def main():
    # 环境准备
    local_md_dir = "output/ms"
    local_image_dir = local_md_dir + "/images"
    image_dir = str(os.path.basename(local_image_dir))
    os.makedirs(local_image_dir, exist_ok=True)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    # pdf 文件路径
    ms_file_name = "2024少儿编程教育行业发展趋势报告.docx"
    ms_file_path = f"data/{ms_file_name}"
    name_without_suff = ms_file_name.split(".")[0]

    # 创建 Dataset 实例
    ds = read_local_office(ms_file_path)[0]

    # 推理
    infer_result = ds.apply(doc_analyze, ocr=True)
    pipe_result = infer_result.pipe_ocr_mode(image_writer)

    # 导出内容列表文件
    pipe_result.dump_content_list(
        md_writer, f"{name_without_suff}_content_list.json", image_dir)

    # 导出 Markdown 文件
    pipe_result.dump_md(
        md_writer, f"{name_without_suff}.md", image_dir
    )


if __name__ == "__main__":
    main()

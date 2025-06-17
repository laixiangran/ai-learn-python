import os
from typing import Literal
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_office
from magic_pdf.data.read_api import read_local_images


class MinerUParser:
    """
    基于 MinerU 的文件解析器
    """

    def __init__(
        self,
        dump_middle_file=True,  # 是否保存中间文件
        is_ocr=True,  # 是否开启 OCR 模式
    ):
        self.dump_middle_file = dump_middle_file
        self.is_ocr = is_ocr

    def convert_pdf(
        self,
        file_path,
        output_base_dir="outputs/pdf",
        output_file_name="mineru_pdf_test",
    ):
        # 环境准备
        output_image_dir = output_base_dir + "/images"
        image_dir = str(os.path.basename(output_image_dir))
        os.makedirs(output_image_dir, exist_ok=True)
        image_writer, md_writer = FileBasedDataWriter(
            output_image_dir
        ), FileBasedDataWriter(output_base_dir)

        # 读取 pdf 内容
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(file_path)

        # 创建 Dataset 实例
        ds = PymuDocDataset(pdf_bytes)

        # 推理
        if self.is_ocr == True:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # 导出 Markdown 文件
        pipe_result.dump_md(md_writer, f"{output_file_name}.md", image_dir)

        if self.dump_middle_file == True:
            # 在每一页上绘制模型结果
            infer_result.draw_model(
                os.path.join(output_base_dir, f"{output_file_name}_model.pdf")
            )

            # 在每一页上绘制布局结果。
            pipe_result.draw_layout(
                os.path.join(output_base_dir, f"{output_file_name}_layout.pdf")
            )

            # 在每一页上绘制文本块（或文字区域）结果
            pipe_result.draw_span(
                os.path.join(output_base_dir, f"{output_file_name}_spans.pdf")
            )

            # 导出内容列表文件
            pipe_result.dump_content_list(
                md_writer, f"{output_file_name}_content_list.json", image_dir
            )

            # 导出中间 JSON 文件
            pipe_result.dump_middle_json(md_writer, f"{output_file_name}_middle.json")

    def convert_ms_office(
        self,
        file_path,
        output_base_dir="outputs/ms_office",
        output_file_name="mineru_ms_office_test",
    ):
        # 环境准备
        output_image_dir = output_base_dir + "/images"
        image_dir = str(os.path.basename(output_image_dir))
        os.makedirs(output_image_dir, exist_ok=True)
        image_writer, md_writer = FileBasedDataWriter(
            output_image_dir
        ), FileBasedDataWriter(output_base_dir)

        # 创建 Dataset 实例
        ds = read_local_office(file_path)[0]

        # 推理
        if self.is_ocr == True:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # 导出 Markdown 文件
        pipe_result.dump_md(md_writer, f"{output_file_name}.md", image_dir)

        if self.dump_middle_file == True:
            # 在每一页上绘制模型结果
            infer_result.draw_model(
                os.path.join(output_base_dir, f"{output_file_name}_model.pdf")
            )

            # 在每一页上绘制布局结果
            pipe_result.draw_layout(
                os.path.join(output_base_dir, f"{output_file_name}_layout.pdf")
            )

            # 在每一页上绘制文本块（或文字区域）结果
            pipe_result.draw_span(
                os.path.join(output_base_dir, f"{output_file_name}_spans.pdf")
            )

            # 导出内容列表文件
            pipe_result.dump_content_list(
                md_writer, f"{output_file_name}_content_list.json", image_dir
            )

            # 导出中间 JSON 文件
            pipe_result.dump_middle_json(md_writer, f"{output_file_name}_middle.json")

    def convert_image(
        self,
        file_path,
        output_base_dir="outputs/image",
        output_file_name="mineru_image_test",
    ):
        # 环境准备
        output_image_dir = output_base_dir + "/images"
        image_dir = str(os.path.basename(output_image_dir))
        os.makedirs(output_image_dir, exist_ok=True)
        image_writer, md_writer = FileBasedDataWriter(
            output_image_dir
        ), FileBasedDataWriter(output_base_dir)

        # 图片文件路径
        if os.path.isfile(file_path) == True:
            # 创建 Dataset 实例
            ds = read_local_images(file_path)[0]

            if self.is_ocr == True:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            # 导出 Markdown 文件
            pipe_result.dump_md(md_writer, f"{output_file_name}.md", image_dir)

            if self.dump_middle_file == True:
                # 在每一页上绘制模型结果
                infer_result.draw_model(
                    os.path.join(output_base_dir, f"{output_file_name}_model.pdf")
                )

                # 在每一页上绘制布局结果
                pipe_result.draw_layout(
                    os.path.join(output_base_dir, f"{output_file_name}_layout.pdf")
                )

                # 在每一页上绘制文本块（或文字区域）结果
                pipe_result.draw_span(
                    os.path.join(output_base_dir, f"{output_file_name}_spans.pdf")
                )

                # 导出内容列表文件
                pipe_result.dump_content_list(
                    md_writer, f"{output_file_name}_content_list.json", image_dir
                )

                # 导出中间 JSON 文件
                pipe_result.dump_middle_json(
                    md_writer, f"{output_file_name}_middle.json"
                )
        else:
            # 创建 Dataset 实例
            dss = read_local_images(file_path)
            count = 0
            for ds in dss:
                if self.is_ocr == True:
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)

                # 导出 Markdown 文件
                pipe_result.dump_md(md_writer, f"{count}.md", image_dir)

                if self.dump_middle_file == True:
                    # 在每一页上绘制模型结果
                    infer_result.draw_model(
                        os.path.join(output_base_dir, f"{count}_model.pdf")
                    )

                    # 在每一页上绘制布局结果
                    pipe_result.draw_layout(
                        os.path.join(output_base_dir, f"{count}_layout.pdf")
                    )

                    # 在每一页上绘制文本块（或文字区域）结果
                    pipe_result.draw_span(
                        os.path.join(output_base_dir, f"{count}_spans.pdf")
                    )

                    # 导出内容列表文件
                    pipe_result.dump_content_list(
                        md_writer, f"{count}_content_list.json", image_dir
                    )

                    # 导出中间 JSON 文件
                    pipe_result.dump_middle_json(md_writer, f"{count}_middle.json")

                    count += 1


def FileParser(
    dump_middle_file=True,
    is_ocr=True,
    provider: Literal["minerUParser"] = "minerUParser",
):
    if provider == "minerUParser":
        return MinerUParser(dump_middle_file=dump_middle_file, is_ocr=is_ocr)
    else:
        raise ValueError(f"Provider {provider} not supported.")

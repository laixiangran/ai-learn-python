import copy
import json
import os
from pathlib import Path
from typing import Literal
from loguru import logger

# PPStructureV3
from paddleocr import PPStructureV3

# MinerU
from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    prepare_env,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

# Docling
from docling.document_converter import DocumentConverter


def save_to_markdown(output_dir, path, md):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件路径
    output_file = os.path.join(output_dir, os.path.basename(path).split(".")[0] + ".md")

    # 写入文件，使用上下文管理器自动处理资源
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md)
    except (IOError, OSError) as e:
        raise RuntimeError(f"写入文件 {output_file} 时发生错误: {e}") from e


class MinerUParser:
    """
    基于 MinerU（2.1.0）的文件解析器
    """

    def __init__(
        self,
        dump_middle_file=True,  # 是否保存中间文件
    ):
        self.dump_middle_file = dump_middle_file

    def do_parse(
        self,
        output_dir,  # Output directory for storing parsing results
        pdf_file_names: list[str],  # List of PDF file names to be parsed
        pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
        p_lang_list: list[
            str
        ],  # List of languages for each PDF, default is 'ch' (Chinese)
        backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
        parse_method="auto",  # The method for parsing PDF, default is 'auto'
        formula_enable=True,  # Enable formula parsing
        table_enable=True,  # Enable table parsing
        server_url=None,  # Server URL for vlm-sglang-client backend
        f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
        f_draw_span_bbox=True,  # Whether to draw span bounding boxes
        f_dump_md=True,  # Whether to dump markdown files
        f_dump_middle_json=True,  # Whether to dump middle JSON files
        f_dump_model_output=True,  # Whether to dump model output files
        f_dump_orig_pdf=True,  # Whether to dump original PDF files
        f_dump_content_list=True,  # Whether to dump content list files
        f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
    ):
        f_draw_layout_bbox = self.dump_middle_file
        f_draw_span_bbox = self.dump_middle_file
        f_dump_middle_json = self.dump_middle_file
        f_dump_model_output = self.dump_middle_file
        f_dump_orig_pdf = self.dump_middle_file
        f_dump_content_list = self.dump_middle_file
        if backend == "pipeline":
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, start_page_id, end_page_id
                )
                pdf_bytes_list[idx] = new_pdf_bytes

            (
                infer_results,
                all_image_lists,
                all_pdf_docs,
                lang_list,
                ocr_enabled_list,
            ) = pipeline_doc_analyze(
                pdf_bytes_list,
                p_lang_list,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )

            for idx, model_list in enumerate(infer_results):
                model_json = copy.deepcopy(model_list)
                pdf_file_name = pdf_file_names[idx]
                local_image_dir, local_md_dir = prepare_env(
                    output_dir, pdf_file_name, parse_method
                )
                image_writer, md_writer = FileBasedDataWriter(
                    local_image_dir
                ), FileBasedDataWriter(local_md_dir)

                images_list = all_image_lists[idx]
                pdf_doc = all_pdf_docs[idx]
                _lang = lang_list[idx]
                _ocr_enable = ocr_enabled_list[idx]
                middle_json = pipeline_result_to_middle_json(
                    model_list,
                    images_list,
                    pdf_doc,
                    image_writer,
                    _lang,
                    _ocr_enable,
                    formula_enable,
                )

                pdf_info = middle_json["pdf_info"]

                pdf_bytes = pdf_bytes_list[idx]
                if f_draw_layout_bbox:
                    draw_layout_bbox(
                        pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                    )

                if f_draw_span_bbox:
                    draw_span_bbox(
                        pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                    )

                if f_dump_orig_pdf:
                    md_writer.write(
                        f"{pdf_file_name}_origin.pdf",
                        pdf_bytes,
                    )

                if f_dump_md:
                    image_dir = str(os.path.basename(local_image_dir))
                    md_content_str = pipeline_union_make(
                        pdf_info, f_make_md_mode, image_dir
                    )
                    md_writer.write_string(
                        f"{pdf_file_name}.md",
                        md_content_str,
                    )

                if f_dump_content_list:
                    image_dir = str(os.path.basename(local_image_dir))
                    content_list = pipeline_union_make(
                        pdf_info, MakeMode.CONTENT_LIST, image_dir
                    )
                    md_writer.write_string(
                        f"{pdf_file_name}_content_list.json",
                        json.dumps(content_list, ensure_ascii=False, indent=4),
                    )

                if f_dump_middle_json:
                    md_writer.write_string(
                        f"{pdf_file_name}_middle.json",
                        json.dumps(middle_json, ensure_ascii=False, indent=4),
                    )

                if f_dump_model_output:
                    md_writer.write_string(
                        f"{pdf_file_name}_model.json",
                        json.dumps(model_json, ensure_ascii=False, indent=4),
                    )

                logger.info(f"local output dir is {local_md_dir}")
        else:
            if backend.startswith("vlm-"):
                backend = backend[4:]

            f_draw_span_bbox = False
            parse_method = "vlm"
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                pdf_file_name = pdf_file_names[idx]
                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, start_page_id, end_page_id
                )
                local_image_dir, local_md_dir = prepare_env(
                    output_dir, pdf_file_name, parse_method
                )
                image_writer, md_writer = FileBasedDataWriter(
                    local_image_dir
                ), FileBasedDataWriter(local_md_dir)
                middle_json, infer_result = vlm_doc_analyze(
                    pdf_bytes,
                    image_writer=image_writer,
                    backend=backend,
                    server_url=server_url,
                )

                pdf_info = middle_json["pdf_info"]

                if f_draw_layout_bbox:
                    draw_layout_bbox(
                        pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                    )

                if f_draw_span_bbox:
                    draw_span_bbox(
                        pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                    )

                if f_dump_orig_pdf:
                    md_writer.write(
                        f"{pdf_file_name}_origin.pdf",
                        pdf_bytes,
                    )

                if f_dump_md:
                    image_dir = str(os.path.basename(local_image_dir))
                    md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                    md_writer.write_string(
                        f"{pdf_file_name}.md",
                        md_content_str,
                    )

                if f_dump_content_list:
                    image_dir = str(os.path.basename(local_image_dir))
                    content_list = vlm_union_make(
                        pdf_info, MakeMode.CONTENT_LIST, image_dir
                    )
                    md_writer.write_string(
                        f"{pdf_file_name}_content_list.json",
                        json.dumps(content_list, ensure_ascii=False, indent=4),
                    )

                if f_dump_middle_json:
                    md_writer.write_string(
                        f"{pdf_file_name}_middle.json",
                        json.dumps(middle_json, ensure_ascii=False, indent=4),
                    )

                if f_dump_model_output:
                    model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                    md_writer.write_string(
                        f"{pdf_file_name}_model_output.txt",
                        model_output,
                    )

                logger.info(f"local output dir is {local_md_dir}")

    def parse_doc(
        self,
        path_list: list[Path],
        output_dir,
        lang_list=["ch", "en"],
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None,
    ):
        """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
        """
        try:
            file_name_list = []
            pdf_bytes_list = []
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
            self.do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )
        except Exception as e:
            logger.exception(e)


class PPStructureV3Parser:
    """
    基于 PPStructure V3（3.1.0）的文件解析器
    """

    def __init__(self, dump_middle_file=True):
        self.dump_middle_file = dump_middle_file

    def parse_doc(self, path_list, output_dir):
        pipeline = PPStructureV3()
        for path in path_list:
            # ocr = PPStructureV3(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
            # ocr = PPStructureV3(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
            # ocr = PPStructureV3(use_textline_orientation=True) # 通过 use_textline_orientation 指定是否使用文本行方向分类模型
            # ocr = PPStructureV3(device="gpu") # 通过 device 指定模型推理时使用 GPU
            output = pipeline.predict(path)
            for res in output:
                if self.dump_middle_file:
                    ## 保存当前pdf/图像的结构化 json 结果
                    res.save_to_json(save_path=output_dir)
                ## 保存当前pdf/图像的 markdown 格式的结果
                res.save_to_markdown(save_path=output_dir)


class DoclingParser:
    """
    基于 Docling 的文件解析器
    """

    def __init__(self, dump_middle_file=True):
        self.dump_middle_file = dump_middle_file

    def parse_doc(self, path_list, output_dir):
        converter = DocumentConverter()
        for path in path_list:
            result = converter.convert(path)
            md = result.document.export_to_markdown()
            save_to_markdown(output_dir, path, md)


def FileParser(
    dump_middle_file=True,
    provider: Literal["MinerU", "PPStructureV3", "Docling"] = "MinerU",
):
    if provider == "MinerU":
        return MinerUParser(dump_middle_file=dump_middle_file)
    elif provider == "PPStructureV3":
        return PPStructureV3Parser(dump_middle_file=dump_middle_file)
    elif provider == "Docling":
        return DoclingParser(dump_middle_file=dump_middle_file)
    else:
        raise ValueError(f"Provider {provider} not supported.")

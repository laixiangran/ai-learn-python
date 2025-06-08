import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_images


def main(isSingle=True):
    # 环境准备
    local_md_dir = "output/img"
    local_image_dir = local_md_dir + "/images"
    image_dir = str(os.path.basename(local_image_dir))
    os.makedirs(local_image_dir, exist_ok=True)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    # 图片文件路径（单个）
    if isSingle == True:
        img_file_name = "1a912cb2f578accf51f9ea2ec548e2a29be101b09b5d4158717093a39901eb9c.jpg"
        img_file_path = f"data/{img_file_name}"
        name_without_suff = img_file_name.split(".")[0]

        # 创建 Dataset 实例
        ds = read_local_images(img_file_path)[0]

        # 导出 Markdown 文件
        ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
            md_writer, f"{img_file_name}.md", image_dir
        )
    else:
        # 图片文件路径（批量）
        img_file_directory = "data/images/"

        # 创建 Dataset 实例
        # read_local_images(input_directory, suffixes=['.png', '.jpg'])
        dss = read_local_images(img_file_directory)
        count = 0
        for ds in dss:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

            # 导出内容列表文件
            pipe_result.dump_content_list(
                md_writer, f"{count}_content_list.json", image_dir)

            # 导出 Markdown 文件
            pipe_result.dump_md(
                md_writer, f"{count}.md", image_dir
            )

            count += 1


if __name__ == "__main__":
    # 单个图片处理
    # main(True)

    # 批量图片处理
    main(False)

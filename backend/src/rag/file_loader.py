from typing import Union, List, Literal, Optional
import glob
from tqdm import tqdm
import multiprocessing
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_non_utf_characters(text: str) -> str:
    """Loại bỏ các ký tự không phải UTF-8."""
    return ''.join(char for char in text if ord(char) < 128)



def load_pdf(pdf_file: str) -> List:
    """
    Tải một file pdf và xử lý nội dung.

    Args:
        pdf_file: Đường dẫn đến file PDF

    Return:
        List: Danh sách các document từ PDF
    """
    try:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(Path(pdf_file).absolute())  # Lưu full path
            doc.metadata["title"] = Path(pdf_file).stem  # Tên file làm title
        return docs
    except Exception as e:
        logger.error(f"Không thể tải file {pdf_file}: {str(e)}")

def get_num_cpu() -> int:
    """Lấy số lượng CPU có sẵn."""
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self):
        self.num_cpu_process = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        """Phương thức gọi loader"""
        pass

class PDFLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        """
        Tải nhiều file PDF với xử lý đa luồng.

        Args:
            pdf_file: Danh sách đường dẫn đến các file PDF
            **kwargs: Tham số bổ sung, bao gồm 'workers' để chỉ định số luồng 

        Returns:
            List: Danh sách tất cả documents từ các PDF
        """
        workers = kwargs.get("workers", 1)
        num_processes = min(self.num_cpu_process, workers)

        if num_processes > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                doc_loaded = []
                total_files = len(pdf_files)
                with tqdm(total=total_files, desc="Đang tải PDF", unit="file") as pbar:
                    for result in pool.imap_unordered(load_pdf, pdf_files):
                        doc_loaded.extend(result)
                        pbar.update(1)
        else:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Đang tải PDF", unit="file") as pbar:
                for pdf_file in pdf_files:
                    result = load_pdf(pdf_file)
                    doc_loaded.extend(result)
                    pbar.update(1)
        
        logger.info(f"Đã tải {len(doc_loaded)} trang từ file PDF")
        return doc_loaded
    
class TextSplitter:
    """Phân chia văn bản thành các đoạn nhỏ hơn."""
    def __init__(self,
                separators: List[str] = ["\n", " ",",", ".", ";","\n\n"],
                chunk_size: int = 300,
                chunk_overlap: int = 30
                ) -> None:
        """
        Khởi tạo text splitter.

        Args:
            separators: Danh sách các ký tự phân tách
            chunk_size: Kích thước tối đa của mỗi đoạn
            chunk_overlap: Số ký tự chồng lấp giữa các đoạn
        """

        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def __call__(self, documents):
        """
        Phân chia documents thành các đoạn nhỏ hơn.

        Args:
            document: Danh sách các document cần phân chia

        Returns:
            List: Danh sachs các documents đã được phân chia
        """

        if not documents:
            logger.warning("Không có documents để phân chia")
            return []
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Đã phân chua thành {len(chunks)} đoạn văn bản")
        return chunks
        

class Loader:
    """Lớp chính để tải và xử lý các tài liệu"""
    def __init__(self,
                file_type: Literal['pdf'] = 'pdf',
                split_kwargs: Optional[dict] = None
                ) -> None:
        if file_type != 'pdf':
            raise ValueError("Hiện tại chỉ hỗ trợ file PDF")
        
        self.file_type = file_type
        self.doc_loader = PDFLoader()

        #Tham số mặc định cho text splitter
        if split_kwargs is None:
            split_kwargs = {
                "chunk_size": 500,
                "chunk_overlap":100
            }

        self.doc_splitter = TextSplitter(**split_kwargs)


    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        """
        Tải và xử lý các file PDF.

        Args:
            pdf_files: Đường dẫn đến file PDF hoặc danh sách các đường dẫn
            workers: Số luồng xử lý đồng thời

        Returns:
            List: Danh sách các document đã được phân chia
        """

        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        
        if not pdf_files:
            logger.warning("Không có file PDF nào được cung cấp")
            return []
        
        logger.info(f"Bắt đầu tải {len(pdf_files)} file PDF với {workers} luồng")
        doc_loaded = self.doc_loader(pdf_files, workers=workers)

        if not doc_loaded:
            logger.warning("Không có dữ liệu nào được tải")
            return []
        
        doc_split = self.doc_splitter(doc_loaded)
        return doc_split
    
    def load_dir(self, dir_path: str, workers: int = 1):
        """
        Tải tất cả file PDF từ một thư mục.
        
        Args:
            dir_path: Đường dẫn đến thư mục các file PDF
            workers: Số luồng xử lý đồng thời

        Returns:
            List: Danh sách các document đã được phân chia
        """

        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            if not files:
                logger.error(f"Không tìm thấy file nào trong {dir_path}")
                return []
        else:
            raise ValueError("Hiện tại chỉ hỗ trợ file PDF")
        logger.info(f"Tìm thấy {len(files)} file PDF trong thư mục {dir_path}")
        return self.load(files,workers=workers)

# if __name__ == "__main__":
#     try:
#         loader = Loader(split_kwargs={"chunk_size": 1000, "chunk_overlap": 200})
#         documents = loader.load_dir("./data_source/generative_ai/pdfs", workers=8)
#         print(f"Đã tải và xử lý {len(documents)} đoạn văn bản")
        
#     except Exception as e:
#         logger.error(f"Lỗi khi chạy ứng dụng: {str(e)}")

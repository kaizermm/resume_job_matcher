from pathlib import Path
from pypdf import PdfReader
import docx2txt

RESUME_DIR = Path("data/resume")

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text).strip()

    if suffix == ".docx":
        return (docx2txt.process(str(file_path)) or "").strip()

    raise ValueError("Resume must be a .pdf or .docx file")

def main():
    # pick resume file automatically (pdf first, then docx)
    pdf = RESUME_DIR / "resume.pdf"
    docx = RESUME_DIR / "resume.docx"

    if pdf.exists():
        resume_path = pdf
    elif docx.exists():
        resume_path = docx
    else:
        raise FileNotFoundError(
            "Put your resume at data/resume/resume.pdf OR data/resume/resume.docx"
        )

    text = extract_text(resume_path)

    # basic cleanup
    text = " ".join(text.split())  # collapse whitespace into single spaces

    out_path = RESUME_DIR / "resume.txt"
    out_path.write_text(text, encoding="utf-8")

    print(f"Read:  {resume_path}")
    print(f"Saved: {out_path}")
    print("Preview:", text[:300])

if __name__ == "__main__":
    main()

from semantic_splitter import SemanticSplitter

def test_auto_split():
    splitter = SemanticSplitter(threshold=0.4, depth='standard', tokenization_mode='para')

    with open(r"C:\Users\DELL\Downloads\maran.txt", "r", encoding="utf-8") as f:
        doc = f.read()

    chunks = splitter.auto_split(doc)

    print(f"Generated {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")

    # You can add assertions to make it a real test, for example:
    assert len(chunks) > 0
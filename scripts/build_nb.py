"""Helper to build .ipynb files from a simple list of (celltype, source) tuples."""
import nbformat as nbf


def build_notebook(cells, out_path):
    nb = nbf.v4.new_notebook()
    nb_cells = []
    for ctype, source in cells:
        source = source.strip("\n")
        if ctype == "md":
            nb_cells.append(nbf.v4.new_markdown_cell(source))
        elif ctype == "code":
            nb_cells.append(nbf.v4.new_code_cell(source))
        else:
            raise ValueError(ctype)
    nb["cells"] = nb_cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", out_path)

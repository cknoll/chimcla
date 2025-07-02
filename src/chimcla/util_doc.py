import os
import subprocess
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()


# created by llama-4-Scout (+ manual adaptions)
def generate_module_docs(package_path: str|None = None, output_dir: str|None = None):

    if package_path is None:
        package_path = os.path.abspath(os.path.dirname(__file__))

    package_name = "chimcla"

    assert os.path.isdir(package_path)
    assert os.path.isfile(Path(package_path)/"__init__.py")
    if output_dir is None:
        repo_root = os.path.dirname(os.path.dirname(package_path))
        output_dir = Path(repo_root)/"doc"/"source"
        module_dir = output_dir/"modules"

    import pkgutil
    index_path = os.path.join(output_dir, 'api_links.md')

    os.makedirs(module_dir, exist_ok=True)

    with open(index_path, 'w') as index_file:
        index_file.write('# Module Documentation\n\n')
        for module_info in pkgutil.walk_packages([package_path]):
            module_name = module_info.name

            index_file.write(f"* [{module_name}](apidocs/{package_name}/{package_name}.{module_name}.md)\n")

            # print(f"File created: {module_path}")
        print(f"File created: {index_path}")


def make_html_doc():
    subprocess.run(['sphinx-build', '-b', 'html', 'doc/source', 'doc/build'])


class DocumentationGenerator(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            make_html_doc()

def continuously_build_docs():
    """
    Watch file system changes (by polling) and trigger build on changes.

    Use CTRL-C to cancel.
    """
    event_handler = DocumentationGenerator()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
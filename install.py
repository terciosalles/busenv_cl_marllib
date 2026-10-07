import os
import shutil
import sys
from pathlib import Path

def merge_directories(source_dir, target_dir):
    """
    Copia todos os arquivos de source_dir para target_dir.
    Adiciona novos arquivos e sobrescreve os existentes.
    NENHUM arquivo existente no destino será apagado.
    """
    if not source_dir.exists():
        return 0

    files_patched = 0
    
    for source_file in source_dir.rglob('*'):
        if source_file.is_dir():
            continue
            
        # Ignora arquivos de cache do Python
        if source_file.suffix in ['.pyc', '.pyo'] or "__pycache__" in source_file.parts:
            continue

        # Calcula o caminho relativo para espelhar a estrutura no destino
        relative_path = source_file.relative_to(source_dir)
        dest_file = target_dir / relative_path
        
        # Cria a pasta de destino caso ela não exista
        if not dest_file.parent.exists():
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            print(f" Nova pasta criada: {dest_file.parent}")

        try:
            # Lógica especial para o __init__.py (injeção de código)
            if source_file.name == "__init__.py" and dest_file.exists():
                with open(source_file, "r", encoding="utf-8") as f:
                    new_content = f.read()
                
                if "run_cl" in new_content:
                    with open(dest_file, "r", encoding="utf-8") as f:
                        old_content = f.read()
                    
                    if "run_cl" not in old_content:
                        print(f" Injetando import no __init__.py: {relative_path}")
                        with open(dest_file, "a", encoding="utf-8") as f:
                            f.write("\n" + new_content)
                        files_patched += 1
                        continue # Pula para o próximo arquivo para não fazer o shutil.copy2
            
            # Cópia padrão (sobrescreve se existir, adiciona se for novo)
            shutil.copy2(source_file, dest_file)
            print(f" Atualizado/Adicionado: {relative_path}")
            files_patched += 1

        except Exception as e:
            print(f" Falha ao copiar {relative_path}: {e}")
            
    return files_patched


def install_patch():
    print("Starting the MARLlib patch installer (MCRL/Continual Learning)...")

    # 1. Encontrar o diretório de instalação do MARLlib
    try:
        import marllib
        target_install_dir = Path(marllib.__file__).parent
        print(f"Target installation found in: {target_install_dir}")
    except ImportError:
        print("  ERROR: The original 'marllib' library was not found.")
        print("  Make sure to install the original marllib first. (pip install .)")
        sys.exit(1)

    installer_dir = Path(__file__).parent
    total_files_patched = 0

    # 2. Mesclar a pasta 'marllib'
    patch_source_pkg = installer_dir / "marllib" 
    if not patch_source_pkg.exists():
        print(f" ERROR: Source folder 'marllib' not found in: {installer_dir}")
        print("  Check if the folder structure is correct.")
        sys.exit(1)

    print(f"\nReading modifications from: {patch_source_pkg}")
    total_files_patched += merge_directories(patch_source_pkg, target_install_dir)

    # 3. Mesclar a pasta 'pipeline'
    # Esta parte agora pega os arquivos de instalador/pipeline e os move
    # para a instalação do marllib (marllib/pipeline).
    pipeline_source = installer_dir / "pipeline"
    if pipeline_source.exists():
        print(f"\nReading modifications from: {pipeline_source}")
        target_pipeline_dir = target_install_dir / "pipeline"
        total_files_patched += merge_directories(pipeline_source, target_pipeline_dir)

    # Finalização
    print("-" * 50)
    print(f" Patch successfully applied! {total_files_patched} arquivos atualizados/adicionados.")
    
    # Aviso para a pasta 'tests' (se existir) que continuará sem ser copiada
    extra_dirs = ["tests"]
    found_extras = [d for d in extra_dirs if (installer_dir / d).exists()]
    if found_extras:
        print("\n  NOTE: The extra folder found in the package ('tests')")
        print("  It was NOT copied to the system, as it is not part of the installed library.")
        print("  If you need it, manually copy it to your project folder.")

    input("\nPress ENTER to exit...")

if __name__ == "__main__":
    install_patch()
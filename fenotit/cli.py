"""Línea de comandos: sin argumentos abre la interfaz; con una ruta .fenotit abre ese proyecto."""
import argparse

from fenotit import __version__, log


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fenotit", description="FenotIT: digital phenotyping from images")
    parser.add_argument("--version", action="version", version=f"FenotIT {__version__}")
    parser.add_argument("project", nargs="?", help=".fenotit project to open")
    args = parser.parse_args(argv)

    log.setup()
    from fenotit.gui.app import main as run_gui
    run_gui(args.project)

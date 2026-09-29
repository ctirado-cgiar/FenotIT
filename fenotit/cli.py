import argparse

from fenotit import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fenotit", description="FenotIT - fenotipado digital por imágenes")
    parser.add_argument("--version", action="version", version=f"FenotIT {__version__}")
    parser.parse_args(argv)

    from fenotit.gui.app import main as run_gui
    run_gui()

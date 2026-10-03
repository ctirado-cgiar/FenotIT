from fenotit.cli import main

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()      # lote en procesos dentro del .exe (PyInstaller)
    main()

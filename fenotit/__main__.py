from fenotit.cli import main

if __name__ == "__main__":            # los procesos del lote importan este módulo sin abrir la app
    import multiprocessing
    multiprocessing.freeze_support()
    main()

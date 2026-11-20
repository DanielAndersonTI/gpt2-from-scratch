"""Entrada da execução noturna.

É o mesmo `src.train`, com o nome que aparece no protocolo de execução: um
comando por sessão, que lê o último checkpoint, continua de onde parou e para
antes do limite da sessão gratuita do Colab.

    python -m src.run_session --config configs/e1_vanilla.yaml --max-hours 3
"""

from src.train import main

if __name__ == "__main__":
    main()

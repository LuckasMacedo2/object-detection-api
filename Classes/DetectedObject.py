class DetectedObject():
    def __init__(self, retangulo = [], defeituoso = '', nivelDefeito = 0, classe = '', percentualClasse = 0, altura = 0, largura = 0):
        self.retangulo = [int(x) for x in retangulo]
        self.defeituoso = defeituoso
        self.nivelDefeito = nivelDefeito
        self.classe = classe
        self.percentualClasse = percentualClasse
        self.altura = altura
        self.largura = largura

    def __str__(self):
        properties = [
            f"\n",
            f"retangulo = {self.retangulo}",
            f"defeituoso = '{self.defeituoso}'",
            f"nivelDefeito = {self.nivelDefeito}",
            f"classe = '{self.classe}'",
            f"percentualClasse = {self.percentualClasse}",
            f"altura = {self.altura}",
            f"largura = {self.largura}"
        ]
        return '\n'.join(properties)
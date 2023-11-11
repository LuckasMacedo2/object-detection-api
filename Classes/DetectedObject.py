class DetectedObject():
    def __init__(self, retangulo = [], defeituoso = '', nivelDefeito = 0, classe = '', percentualClasse = 0):
        self.retangulo = [int(x) for x in retangulo]
        self.defeituoso = defeituoso
        self.nivelDefeito = nivelDefeito
        self.classe = classe
        self.percentualClasse = percentualClasse
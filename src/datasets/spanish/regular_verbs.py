from transformers import AutoTokenizer

import sys
sys.path.append('../../')  

from src.datasets.spanish.spanish_verbs import spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs
from jsalt2025.src.utils.spanish_dataset_generation import create_spanish_verbs, filter_spanish_conjugations

spanish_verbs = [
    "abandonar", "abordar", "abortar", "aburrir", "aburrirse", "abusar", "acabar", "acampar", "aceptar",
    "acercarse", "acompañar", "aconsejar", "acortar", "acostumbrar", "acostumbrarse", "adivinar", "admirar",
    "admitir", "adorar", "adornar", "afeitar", "afeitarse", "afirmar", "agradar", "aguantar", "ahorrar",
    "alegrar", "alegrarse", "aliviar", "alquilar", "amar", "añadir", "anhelar", "anunciar", "aplaudir",
    "apoyar", "apreciar", "aprender", "arreglar", "arrojar", "asistir", "asociar", "aspirar", "asustar",
    "asustarse", "atreverse", "aumentar", "averiguar", "avisar", "ayudar", "bailar", "bajar", "bañar",
    "bañarse", "barrer", "batir", "beber", "besar", "bordar", "borrar", "brillar", "brindar", "broncearse",
    "bucear", "burlar", "burlarse", "calcular", "callar", "callarse", "calmar", "calmarse", "caminar",
    "cancelar", "cansar", "cansarse", "casar", "casarse", "causar", "celebrar", "cenar", "censurar",
    "cepillar", "cesar", "charlar", "chismear", "cobrar", "cocinar", "coger", "combatir", "comer", "compartir",
    "comprar", "comprender", "condenar", "confirmar", "conquistar", "conservar", "consistir", "constituir",
    "construir", "consumir", "contaminar", "contestar", "contribuir", "controlar", "convencer", "conversar",
    "convidar", "copiar", "correr", "cortar", "coser", "crear", "cubrir", "cuidar", "culpar", "cultivar",
    "cumplir", "curar", "deber", "decidir", "decidirse", "declarar", "decorar", "dedicarse", "dejar",
    "depender", "depositar", "deprimir", "desagradar", "desarrollar", "desarrollarse", "desayunar",
    "descansar", "describir", "descubrir", "desear", "destruir", "detestar", "devorar", "dibujar", "diseñar",
    "disfrutar", "disgustar", "disminuir", "distinguir", "distribuir", "divorciar", "divorciarse", "doblar",
    "duchar", "ducharse", "dudar", "durar", "echar", "echarse", "ejercer", "eliminar", "emborrachar",
    "emborracharse", "emigrar", "emplear", "enamorar", "enamorarse", "encantar", "enfadar", "enfadarse",
    "enfermar", "enfermarse", "engañar", "enojar", "enojarse", "enseñar", "ensuciar", "enterarse", "entrar",
    "entrevistar", "entusiasmar", "entusiasmarse", "errar", "escoger", "esconder", "escribir", "escuchar",
    "esperar", "estimar", "estudiar", "evacuar", "evitar", "exhibir", "explorar", "explotar", "exportar",
    "expresar", "extinguir", "faltar", "fascinar", "felicitar", "fijar", "firmar", "formar", "fumar",
    "funcionar", "ganar", "gastar", "graduarse", "gritar", "gruñir", "guardar", "gustar", "hablar", "hallar",
    "hallarse", "heredar", "huir", "ilustrar", "importar", "imprimir", "incluir", "influir", "informar",
    "iniciar", "inmigrar", "insistir", "instalar", "insultar", "intentar", "interesar", "interpretar",
    "invadir", "inventar", "invitar", "jactarse", "juntar", "juntarse", "jurar", "ladrar", "lamentar",
    "lastimar", "lavar", "lavarse", "levantar", "levantarse", "limpiar", "llamar", "llamarse", "llenar",
    "llevar", "llorar", "lograr", "luchar", "madurar", "mandar", "manejar", "maquillar", "maquillarse",
    "matar", "matricular", "matricularse", "meter", "mezclar", "mirar", "molestar", "montar", "mudar",
    "mudarse", "nadar", "necesitar", "negociar", "notar", "odiar", "ofender", "olvidar", "olvidarse",
    "parar", "participar", "pasar", "patinar", "peinar", "peinarse", "pelear", "perdonar", "permitir",
    "pesar", "pintar", "planchar", "plantar", "preguntar", "preguntarse", "preparar", "prepararse",
    "presentar", "preservar", "prohibir", "prometer", "proteger", "protestar", "quedar", "quedarse",
    "quejarse", "quemar", "quemarse", "recibir", "reciclar", "recoger", "regalar", "regatear", "registrar",
    "registrarse", "regresar", "regular", "rehusar", "reinar", "renunciar", "reparar", "repasar", "reportar",
    "reservar", "respetar", "respirar", "responder", "resultar", "revelar", "robar", "romper", "sacudir",
    "saltar", "saludar", "salvar", "señalar", "sobrevivir", "soportar", "sorprender", "subir", "suceder",
    "sufrir", "suspirar", "sustituir", "tañer", "tapar", "tardar", "temer", "terminar", "tirar", "tomar",
    "toser", "trabajar", "tratar", "triunfar", "unir", "untar", "usar", "vaciar", "vencer", "vender",
    "viajar", "violar", "visitar", "vivir", "vomitar", "votar", "zambullirse"
]


# Conjugate all regular verbs assuming they are regular based on ending
def conjugate_regular(verb):
    if verb.endswith("ar"):
        stem = verb[:-2]
        return (verb, stem + "o", stem + "as", "ar", "reg")
    elif verb.endswith("er"):
        stem = verb[:-2]
        return (verb, stem + "o", stem + "es", "er", "reg")
    elif verb.endswith("ir"):
        stem = verb[:-2]
        return (verb, stem + "o", stem + "es", "ir", "reg")
    else:
        return (verb, None, None, "?", "unknown")

# Apply conjugation
full_conjugated_list = [conjugate_regular(verb) for verb in spanish_verbs]

# Count valid (regular) conjugations
valid_conjugated = [entry for entry in full_conjugated_list if entry[3] in ["ar", "er", "ir"]]

print("original list length: ",len(spanish_verbs))
print("filtered list length: ", len(full_conjugated_list))
print("filtered list length: ", len(valid_conjugated))
print("PRINT:", valid_conjugated)
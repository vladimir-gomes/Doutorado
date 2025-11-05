import rasterio
import numpy as np
import pandas as pd

def read_wavelengths_from_csv(filepath):
    """Lê comprimentos de onda de um arquivo CSV simples."""
    return pd.read_csv(filepath).iloc[:, 0].values

class SatelliteImage:
    """
    Uma classe para representar uma imagem de satélite hiperespectral.
    """
    def __init__(self, image_path, wavelength_source):
        """
        Inicializa o objeto de imagem de satélite.

        Args:
            image_path (str): Caminho para o arquivo de imagem (ex: GeoTIFF).
            wavelength_source (str or list): Caminho para o arquivo de comprimentos de onda (CSV)
                                             ou uma lista/array de comprimentos de onda.
        """
        self.image_path = image_path
        self.src = rasterio.open(image_path)
        self.wavelengths = self._load_wavelengths(wavelength_source)

        if self.src.count != len(self.wavelengths):
            raise ValueError(
                f"O número de bandas na imagem ({self.src.count}) não corresponde ao "
                f"número de comprimentos de onda fornecidos ({len(self.wavelengths)})."
            )

    def _load_wavelengths(self, source):
        """Carrega os comprimentos de onda de uma fonte especificada."""
        if isinstance(source, str):
            if source.lower().endswith('.csv'):
                print(f"Carregando comprimentos de onda de: {source}")
                return read_wavelengths_from_csv(source)
            else:
                # Adicionar lógica para outros formatos de metadados (ex: XML para EnMAP)
                raise NotImplementedError(f"Formato de arquivo de comprimento de onda não suportado: {source}")
        elif isinstance(source, (list, np.ndarray)):
            return np.array(source)
        else:
            raise TypeError("A fonte de comprimentos de onda deve ser um caminho de arquivo ou uma lista/array.")

    def extract_signatures(self, points_df, lat_col='latitude', lon_col='longitude'):
        """
        Extrai assinaturas espectrais de uma lista de pontos.

        Args:
            points_df (pd.DataFrame): DataFrame com as coordenadas dos pontos.
            lat_col (str): Nome da coluna de latitude.
            lon_col (str): Nome da coluna de longitude.

        Returns:
            pd.DataFrame: Um DataFrame com as assinaturas espectrais extraídas.
        """
        coords = list(zip(points_df[lon_col], points_df[lat_col]))

        # rasterio.sample.sample_gen retorna um gerador
        # O resultado é uma lista de arrays numpy, um para cada ponto
        sampled_data_gen = self.src.sample(coords)

        # Converte o gerador em uma lista e depois em um DataFrame
        signatures_list = list(sampled_data_gen)
        signatures_array = np.vstack(signatures_list)

        # Cria nomes de coluna baseados nos comprimentos de onda
        column_names = [f'reflectance_{wl:.2f}' for wl in self.wavelengths]

        return pd.DataFrame(signatures_array, columns=column_names, index=points_df.index)

# --- Funções de conveniência para diferentes sensores ---

def load_emit_image(image_path, wavelengths_path):
    """
    Carrega uma imagem EMIT.
    Assume que os comprimentos de onda são fornecidos em um arquivo CSV.
    """
    print("Inicializando imagem EMIT...")
    return SatelliteImage(image_path, wavelengths_path)

def load_enmap_image(image_path, metadata_xml_path):
    """
    Carrega uma imagem EnMAP.
    Esta é uma função placeholder. A lógica para extrair comprimentos de onda
    do arquivo XML de metadados precisaria ser implementada aqui.
    """
    print(f"Inicializando imagem EnMAP (lógica de metadados a ser implementada)...")
    # LÓGICA PLACEHOLDER:
    # 1. Parsear o arquivo metadata_xml_path.
    # 2. Extrair a lista de comprimentos de onda.
    # 3. Retornar o objeto SatelliteImage.
    # Exemplo com comprimentos de onda falsos:
    dummy_wavelengths = np.linspace(420, 2450, 224) # Exemplo para EnMAP
    print("AVISO: Usando comprimentos de onda placeholder para EnMAP.")
    return SatelliteImage(image_path, dummy_wavelengths)

def load_prisma_image(image_path, wavelengths_path):
    """
    Carrega uma imagem PRISMA.
    Assume que os comprimentos de onda são fornecidos em um arquivo CSV.
    """
    print("Inicializando imagem PRISMA...")
    return SatelliteImage(image_path, wavelengths_path)

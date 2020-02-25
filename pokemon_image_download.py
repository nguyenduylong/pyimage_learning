from bing_image_api_using import PokemonImageDownload
import argparse
import os

pokemons_names = [ 'Charmander', 'Pikachu', 'Bulbasaur', 'Squirtle', 'Caterpie', 'Butterfree', 'Mankey', 'Machop', 'Cubone', 'Koffing', 'Mr. Mime', 'Snorlax']

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())

print(args);

for pokemon_name in pokemons_names:
    save_folder = os.path.join(args["output"], pokemon_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    PokemonImageDownload.download(pokemon_name, save_folder)
from midi_reader import MidiReader
from song_importer import SongImporter
from music_generator import MusicGenerator

coordinator = MidiCoordinator(24, 102)
importer = SongImporter('midi_songs', coordinator)
generator = MusicGenerator(coordinator)

imported_songs = importer.getSongs()
generator.generateSongs(imported_songs)

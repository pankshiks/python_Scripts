import lyricsgenius

genius = lyricsgenius.Genius("hNx-N_zfU6HSt0uXvjtd4Hc8JFHrvGtD7sRzICr8k3GNi5zLE3J8-Vc2ta4Et_6B")
artist = genius.search_artist("taylor swift", max_songs=0, sort="title")
song = artist.song("blank space")
with open('lyrics.txt',"w") as f:
    f.write(song.lyrics)
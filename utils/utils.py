import os
import re
import json
import logging
from logging.handlers import BaseRotatingHandler
from colorama import Fore

from langchain.agents.agent_toolkits.openapi.spec import ReducedOpenAPISpec



class ColorPrint:
    def __init__(self):
        self.color_mapping = {
            "Planner": Fore.RED,
            "API Selector": Fore.YELLOW,
            "Caller": Fore.BLUE,
            "Parser": Fore.GREEN,
            "Code": Fore.WHITE,
        }

    def write(self, data):
        module = data.split(':')[0]
        if module not in self.color_mapping:
            print(data, end="")
        else:
            print(self.color_mapping[module] + data + Fore.RESET, end="")


class MyRotatingFileHandler(BaseRotatingHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        BaseRotatingHandler.__init__(self, filename, mode, encoding, delay)
        self.cnt = 1

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        
        dfn = self.rotation_filename('.'.join(self.baseFilename.split('.')[:-1]) + f"_{self.cnt}." + self.baseFilename.split('.')[-1])
        if os.path.exists(dfn):
            os.remove(dfn)
        self.rotate(self.baseFilename, dfn)
        self.cnt += 1
        
        if not self.delay:
            self.stream = self._open()

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()
        return 0


def get_matched_endpoint(api_spec: ReducedOpenAPISpec, plan: str):
    pattern = r"\b(GET|POST|PATCH|DELETE|PUT)\s+(/\S+)*"
    matches = re.findall(pattern, plan)
    plan_endpoints = [
        "{method} {route}".format(method=method, route=route.split("?")[0])
        for method, route in matches
    ]
    spec_endpoints = [item[0] for item in api_spec.endpoints]

    matched_endpoints = []

    for plan_endpoint in plan_endpoints:
        if plan_endpoint in spec_endpoints:
            matched_endpoints.append(plan_endpoint)
            continue
        for name in spec_endpoints:
            arg_list = re.findall(r"[{](.*?)[}]", name)
            pattern = name.format(**{arg: r"[^/]+" for arg in arg_list}) + '$'
            if re.match(pattern, plan_endpoint):
                matched_endpoints.append(name)
                break
    if len(matched_endpoints) == 0:
        return None
        # raise ValueError(f"Endpoint {plan_endpoint} not found in API spec.")
    
    return matched_endpoints


def simplify_json(raw_json: dict):
    if isinstance(raw_json, dict):
        for key in raw_json.keys():
            raw_json[key] = simplify_json(raw_json[key])
        return raw_json
    elif isinstance(raw_json, list):
        if len(raw_json) == 0:
            return raw_json
        elif len(raw_json) == 1:
            return [simplify_json(raw_json[0])]
        else:
            return [simplify_json(raw_json[0]), simplify_json(raw_json[1])]
    else:
        return raw_json


def fix_json_error(data: str, return_str=True):
    data = data.strip().strip('"').strip(",").strip("`")
    try:
        json.loads(data)
        return data
    except json.decoder.JSONDecodeError:
        data = data.split("\n")
        data = [line.strip() for line in data]
        for i in range(len(data)):
            line = data[i]
            if line in ['[', ']', '{', '}']:
                continue
            if line.endswith(('[', ']', '{', '}')):
                continue
            if not line.endswith(',') and data[i + 1] not in [']', '}', '],', '},']:
                data[i] += ','
            if data[i + 1] in [']', '}', '],', '},'] and line.endswith(','):
                data[i] = line[:-1]
        data = " ".join(data)
        
        if not return_str:
            data = json.loads(data)
        return data


def init_spotify(requests_wrapper):
    # WARNING: this will remove all your data from spotify!!!
    # The final environment:
    # Your Music: 6 tracks, top-3 tracks of Lana Del Rey and Whitney Houston
    # Your Albums: Born To Die by Lana Del Rey and reputation by Taylor Swift
    # Your Playlists: My R&B with top-3 tracks of Whitney Houston and My Rock with top-3 tracks of Beatles
    # Your Followed Artists: Lana Del Rey, Whitney Houston, and Beatles
    # Current Playing: album Born To Die by Lana Del Rey

    user_id = json.loads(requests_wrapper.get('https://api.spotify.com/v1/me').text)['id']

    # remove all playlists
    playlist_ids = json.loads(requests_wrapper.get('https://api.spotify.com/v1/me/playlists').text)['items']
    playlist_ids = [playlist['id'] for playlist in playlist_ids]

    for playlist_id in playlist_ids:
        requests_wrapper.delete(f'https://api.spotify.com/v1/playlists/{playlist_id}/followers')

    # remove all tracks from my music
    track_ids = json.loads(requests_wrapper.get('https://api.spotify.com/v1/me/tracks').text)['items']
    track_ids = [track['track']['id'] for track in track_ids]
    if len(track_ids) != 0:
        requests_wrapper.delete(f'https://api.spotify.com/v1/me/tracks?ids={",".join(track_ids)}')

    # remove all albums from my music
    album_ids = json.loads(requests_wrapper.get('https://api.spotify.com/v1/me/albums').text)['items']
    album_ids = [album['album']['id'] for album in album_ids]
    if len(album_ids) != 0:
        requests_wrapper.delete(f'https://api.spotify.com/v1/me/albums?ids={",".join(album_ids)}')

    # remove all following artists
    artist_ids = json.loads(requests_wrapper.get('https://api.spotify.com/v1/me/following?type=artist').text)['artists']['items']
    artist_ids = [artist['id'] for artist in artist_ids]
    if len(artist_ids) != 0:
        requests_wrapper.delete(f'https://api.spotify.com/v1/me/following?type=artist&ids={",".join(artist_ids)}')

    # add top-3 tracks of Lana Del Rey, Whitney Houston to my music
    artist_id_1 = requests_wrapper.get(f'https://api.spotify.com/v1/search?q=Lana%20Del%20Rey&type=artist')
    artist_id_1 = json.loads(artist_id_1.text)['artists']['items'][0]['id']
    track_ids_1 = requests_wrapper.get(f'https://api.spotify.com/v1/artists/{artist_id_1}/top-tracks?country=US')
    track_ids_1 = json.loads(track_ids_1.text)['tracks']
    track_ids_1 = [track['id'] for track in track_ids_1][:3]
    requests_wrapper.put(f'https://api.spotify.com/v1/me/tracks?ids={",".join(track_ids_1)}', data=None)

    artist_id_2 = requests_wrapper.get(f'https://api.spotify.com/v1/search?q=Whitney%20Houston&type=artist')
    artist_id_2 = json.loads(artist_id_2.text)['artists']['items'][0]['id']
    track_ids_2 = requests_wrapper.get(f'https://api.spotify.com/v1/artists/{artist_id_2}/top-tracks?country=US')
    track_ids_2 = json.loads(track_ids_2.text)['tracks']
    track_ids_2 = [track['id'] for track in track_ids_2][:3]
    requests_wrapper.put(f'https://api.spotify.com/v1/me/tracks?ids={",".join(track_ids_2)}', data=None)

    # search for the top-3 tracks of The Beatles
    artist_id_3 = requests_wrapper.get(f'https://api.spotify.com/v1/search?q=The%20Beatles&type=artist')
    artist_id_3 = json.loads(artist_id_3.text)['artists']['items'][0]['id']
    track_ids_3 = requests_wrapper.get(f'https://api.spotify.com/v1/artists/{artist_id_3}/top-tracks?country=US')
    track_ids_3 = json.loads(track_ids_3.text)['tracks']
    track_ids_3 = [track['id'] for track in track_ids_3][:3]

    # follow Lana Del Rey, Whitney Houston, The Beatles
    requests_wrapper.put(f'https://api.spotify.com/v1/me/following?type=artist&ids={",".join([artist_id_1, artist_id_2, artist_id_3])}', data=None)

    # create playlist My R&B, My Rock. Add top-3 tracks of Whitney Houston to My R&B, top-3 tracks of The Beatles to My Rock
    playlist_id_1 = requests_wrapper.post(f'https://api.spotify.com/v1/users/{user_id}/playlists', data={'name': 'My R&B'})
    playlist_id_1 = json.loads(playlist_id_1.text)['id']
    requests_wrapper.post(f'https://api.spotify.com/v1/playlists/{playlist_id_1}/tracks?uris={",".join([f"spotify:track:{track_id}" for track_id in track_ids_2])}', data=None)

    playlist_id_2 = requests_wrapper.post(f'https://api.spotify.com/v1/users/{user_id}/playlists', data={'name': 'My Rock'})
    playlist_id_2 = json.loads(playlist_id_2.text)['id']
    requests_wrapper.post(f'https://api.spotify.com/v1/playlists/{playlist_id_2}/tracks?uris={",".join([f"spotify:track:{track_id}" for track_id in track_ids_3])}', data=None)

    # add Born To Die and reputation to my album. play Lana Del Rey's album "Born To Die"
    album_id_1 = requests_wrapper.get(f'https://api.spotify.com/v1/search?q=Born%20To%20Die&type=album')
    album_id_1 = json.loads(album_id_1.text)['albums']['items'][0]['uri']
    requests_wrapper.put(f'https://api.spotify.com/v1/me/albums?ids={album_id_1}', data=None)
    requests_wrapper.put(f'https://api.spotify.com/v1/me/player/play', data={'context_uri': album_id_1})

    album_id_2 = requests_wrapper.get(f'https://api.spotify.com/v1/search?q=reputation&type=album')
    album_id_2 = json.loads(album_id_2.text)['albums']['items'][0]['uri']
    requests_wrapper.put(f'https://api.spotify.com/v1/me/albums?ids={album_id_2}', data=None)


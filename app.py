from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
import requests
from datetime import datetime
import os
import glob
import tempfile
import time
from dotenv import load_dotenv

load_dotenv()

# Clear any CA bundle env vars so they don't affect third-party API calls (ElevenLabs etc.)
# Revinci uses a self-signed cert so its calls explicitly pass verify=False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.pop('REQUESTS_CA_BUNDLE', None)
os.environ.pop('CURL_CA_BUNDLE', None)
REVINCI_SSL_VERIFY = False  # Revinci dev server uses self-signed cert

import subprocess

# Ensure ffmpeg is on PATH (required for audio conversion)
_ffmpeg_search = glob.glob(
    os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\WinGet\Packages\*FFmpeg*\**\bin'),
    recursive=True
)
for p in _ffmpeg_search:
    if os.path.isdir(p) and p not in os.environ.get('PATH', ''):
        os.environ['PATH'] = p + os.pathsep + os.environ.get('PATH', '')
        print(f"[startup] Added ffmpeg to PATH: {p}")

app = Flask(__name__)
CORS(app)

# Configuration (all secrets loaded from .env)
AZURE_CONFIG = {
    'api_key':    os.getenv('AZURE_OPENAI_API_KEY', ''),
    'endpoint':   os.getenv('AZURE_OPENAI_ENDPOINT', ''),
    'deployment': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', ''),
    'api_version': os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15')
}

WEATHER_CONFIG = {
    'api_key': os.getenv('OPENWEATHERMAP_API_KEY', ''),
    'base_url': 'https://api.openweathermap.org/data/2.5'
}

WIKIPEDIA_CONFIG = {
    'base_url': 'https://en.wikipedia.org/w/api.php',
    'user_agent': 'ConversationalAI/1.0'
}

NEWS_CONFIG = {
    'api_key': os.getenv('NEWS_API_KEY', ''),
    'base_url': 'https://newsapi.org/v2',
    'page_size': 10
}

FINANCIAL_NEWS_CONFIG = {
    'enabled': True,
    'categories': ['business', 'technology'],
    'sources': ['bloomberg', 'financial-times', 'the-wall-street-journal', 'cnbc']
}

# Revinci Backend Chat API (loaded from .env)
REVINCI_CONFIG = {
    'url': os.getenv('REVINCI_API_URL', 'https://api-dev.revinci.ai/ai/opportunity/query'),
    'user_id': os.getenv('REVINCI_USER_ID', ''),
    'keycloak_url': os.getenv('KEYCLOAK_URL', 'https://auth-dev.revinci.ai'),
    'keycloak_realm': os.getenv('KEYCLOAK_REALM', 'tessla'),
    'keycloak_client_id': os.getenv('KEYCLOAK_CLIENT_ID', 'tessla'),
    'auth_username': os.getenv('REVINCI_AUTH_USERNAME', ''),
    'auth_password': os.getenv('REVINCI_AUTH_PASSWORD', ''),
}

# Token cache: {'token': str, 'expires_at': float (unix timestamp)}
_revinci_token_cache = {'token': '', 'expires_at': 0.0}

# ElevenLabs Configuration
ELEVENLABS_CONFIG = {
    'api_key': os.getenv('ELEVENLABS_API_KEY', ''),
    'base_url': 'https://api.elevenlabs.io/v1'
}

# ElevenLabs Voice IDs (popular voices)
ELEVENLABS_VOICES = {
    'female': [
        {'voice_id': '21m00Tcm4TlvDq8ikWAM', 'name': 'Rachel', 'display_name': 'Rachel (Calm)'},
        {'voice_id': 'EXAVITQu4vr4xnSDxMaL', 'name': 'Bella', 'display_name': 'Bella (Soft)'},
        {'voice_id': 'MF3mGyEYCl7XYWbV9V6O', 'name': 'Elli', 'display_name': 'Elli (Energetic)'},
        {'voice_id': 'XrExE9yKIg1WjnnlVkGX', 'name': 'Matilda', 'display_name': 'Matilda (Warm)'},
    ],
    'male': [
        {'voice_id': 'TxGEqnHWrfWFTfGW9XjX', 'name': 'Josh', 'display_name': 'Josh (Professional)'},
        {'voice_id': 'VR6AewLTigWG4xSOukaG', 'name': 'Arnold', 'display_name': 'Arnold (Deep)'},
        {'voice_id': 'pNInz6obpgDQGcFmaJgB', 'name': 'Adam', 'display_name': 'Adam (Narrator)'},
        {'voice_id': 'yoZ06aMxZJJ28mfd3POQ', 'name': 'Sam', 'display_name': 'Sam (Dynamic)'},
    ],
    'other': []
}

conversation_sessions = {}
revinci_conversation_ids = {}  # session_id -> revinci conversation_id

# Azure Speech-to-Text configuration
AZURE_SPEECH_CONFIG = {
    'api_key': os.getenv('AZURE_SPEECH_KEY', ''),
    'region':  os.getenv('AZURE_SPEECH_REGION', 'westus'),
}

SYSTEM_PROMPT = """You are a helpful AI assistant with access to real-time weather information, Wikipedia knowledge, and news articles. 

When users ask about weather, you can provide current conditions, forecasts, and weather-related advice.

When users ask about factual information, you can search Wikipedia for detailed information about people, places, events, and concepts.

When users ask about news, you can provide latest headlines, breaking news, and news about specific topics.

Always provide natural, conversational responses based on the data available."""


def get_weather_data(city):
    try:
        if not WEATHER_CONFIG['api_key']:
            return {'success': False, 'error': 'Weather API key not configured'}

        url = f"{WEATHER_CONFIG['base_url']}/weather"
        params = {
            'q': city,
            'appid': WEATHER_CONFIG['api_key'],
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'temp_min': data['main']['temp_min'],
                'temp_max': data['main']['temp_max'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'main_condition': data['weather'][0]['main'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0),
                'clouds': data['clouds']['all'],
                'visibility': data.get('visibility', 'N/A'),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
                'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
            }
            return {'success': True, 'data': weather_info}
        elif response.status_code == 404:
            return {'success': False, 'error': f'City "{city}" not found'}
        elif response.status_code == 401:
            return {'success': False, 'error': 'Invalid API key'}
        else:
            return {'success': False, 'error': 'Unable to fetch weather data'}
            
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Weather service timeout'}
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Unable to connect to weather service'}
    except Exception as e:
        return {'success': False, 'error': f'Weather API error: {str(e)}'}


def get_weather_forecast(city, days=5):
    try:
        if not WEATHER_CONFIG['api_key']:
            return {'success': False, 'error': 'Weather API key not configured'}

        url = f"{WEATHER_CONFIG['base_url']}/forecast"
        params = {
            'q': city,
            'appid': WEATHER_CONFIG['api_key'],
            'units': 'metric',
            'cnt': min(days * 8, 40)
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            forecast_list = []
            
            for item in data['list'][::8][:days]:
                forecast_list.append({
                    'date': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d'),
                    'day': datetime.fromtimestamp(item['dt']).strftime('%A'),
                    'temperature': item['main']['temp'],
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'description': item['weather'][0]['description'],
                    'humidity': item['main']['humidity'],
                    'wind_speed': item['wind']['speed'],
                    'clouds': item['clouds']['all']
                })
            
            return {'success': True, 'data': forecast_list, 'city': data['city']['name']}
        else:
            return {'success': False, 'error': 'Forecast not available'}
            
    except Exception as e:
        return {'success': False, 'error': f'Forecast API error: {str(e)}'}


def detect_weather_query(message):
    message_lower = message.lower()
    weather_keywords = [
        'weather', 'temperature', 'forecast', 'rain', 'raining', 'rainy',
        'sunny', 'cloudy', 'humidity', 'humid', 'wind', 'windy', 'climate',
        'hot', 'cold', 'warm', 'cool', 'snow', 'snowing', 'storm', 'stormy',
        'celsius', 'fahrenheit', 'degrees', 'umbrella', 'jacket'
    ]
    
    is_weather_query = any(keyword in message_lower for keyword in weather_keywords)
    
    if not is_weather_query:
        return {'is_weather': False}
    
    words = message.split()
    for i, word in enumerate(words):
        if word.lower() in ['in', 'at', 'for', 'of'] and i + 1 < len(words):
            city_parts = []
            for j in range(i + 1, min(i + 4, len(words))):
                word_clean = words[j].rstrip('?,!.')
                if word_clean.lower() not in ['the', 'weather', 'forecast', 'today', 'tomorrow', 'like']:
                    city_parts.append(word_clean)
                else:
                    break
            
            if city_parts:
                return {'is_weather': True, 'city': ' '.join(city_parts)}
    
    return {'is_weather': True, 'city': None}


def search_wikipedia(query, sentences=3):
    try:
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': 1
        }
        
        headers = {'User-Agent': WIKIPEDIA_CONFIG['user_agent']}
        
        search_response = requests.get(
            WIKIPEDIA_CONFIG['base_url'], 
            params=search_params, 
            headers=headers,
            timeout=10
        )
        
        if search_response.status_code != 200:
            return {'success': False, 'error': 'Wikipedia search failed'}
        
        search_data = search_response.json()
        
        if not search_data['query']['search']:
            return {'success': False, 'error': f'No Wikipedia article found for "{query}"'}
        
        page_title = search_data['query']['search'][0]['title']
        
        content_params = {
            'action': 'query',
            'prop': 'extracts|info',
            'exintro': True,
            'explaintext': True,
            'exsentences': sentences,
            'titles': page_title,
            'format': 'json',
            'inprop': 'url'
        }
        
        content_response = requests.get(
            WIKIPEDIA_CONFIG['base_url'],
            params=content_params,
            headers=headers,
            timeout=10
        )
        
        if content_response.status_code != 200:
            return {'success': False, 'error': 'Failed to fetch Wikipedia content'}
        
        content_data = content_response.json()
        pages = content_data['query']['pages']
        page_id = list(pages.keys())[0]
        
        if page_id == '-1':
            return {'success': False, 'error': 'Wikipedia page not found'}
        
        page_info = pages[page_id]
        
        wiki_data = {
            'title': page_info['title'],
            'summary': page_info.get('extract', 'No summary available'),
            'url': page_info.get('fullurl', ''),
            'page_id': page_id
        }
        
        return {'success': True, 'data': wiki_data}
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Wikipedia request timeout'}
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Unable to connect to Wikipedia'}
    except Exception as e:
        return {'success': False, 'error': f'Wikipedia API error: {str(e)}'}


def detect_wikipedia_query(message):
    message_lower = message.lower()
    wikipedia_triggers = [
        'who is', 'who was', 'who are', 'what is', 'what was', 'what are',
        'tell me about', 'information about', 'explain', 'define',
        'wikipedia', 'wiki', 'search for', 'look up',
        'history of', 'biography of', 'facts about'
    ]
    
    is_wiki_query = any(trigger in message_lower for trigger in wikipedia_triggers)
    
    if not is_wiki_query:
        return {'is_wikipedia': False}
    
    for trigger in wikipedia_triggers:
        if trigger in message_lower:
            parts = message_lower.split(trigger, 1)
            if len(parts) > 1:
                search_query = parts[1].strip().rstrip('?.,!')
                if search_query:
                    return {'is_wikipedia': True, 'query': search_query}
    
    return {'is_wikipedia': True, 'query': None}


def get_general_news(query=None, category=None, country='us', page_size=10):
    try:
        if not NEWS_CONFIG['api_key']:
            return {'success': False, 'error': 'News API key not configured'}

        if query:
            url = f"{NEWS_CONFIG['base_url']}/everything"
            params = {
                'q': query,
                'apiKey': NEWS_CONFIG['api_key'],
                'pageSize': min(page_size, NEWS_CONFIG['page_size']),
                'sortBy': 'publishedAt',
                'language': 'en'
            }
        else:
            url = f"{NEWS_CONFIG['base_url']}/top-headlines"
            params = {
                'apiKey': NEWS_CONFIG['api_key'],
                'pageSize': min(page_size, NEWS_CONFIG['page_size']),
                'country': country
            }
            
            if category:
                params['category'] = category
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            error_data = response.json()
            return {'success': False, 'error': error_data.get('message', 'News API request failed')}
        
        data = response.json()
        
        if data['status'] != 'ok':
            return {'success': False, 'error': 'Failed to fetch news'}
        
        if not data.get('articles'):
            return {'success': False, 'error': 'No news articles found'}
        
        articles = []
        for article in data['articles'][:page_size]:
            articles.append({
                'title': article.get('title', 'No title'),
                'description': article.get('description', 'No description'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'author': article.get('author', 'Unknown'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'image_url': article.get('urlToImage', '')
            })
        
        return {
            'success': True,
            'data': {
                'articles': articles,
                'total_results': data.get('totalResults', len(articles)),
                'query': query,
                'category': category
            }
        }
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'News API timeout'}
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Unable to connect to News API'}
    except Exception as e:
        return {'success': False, 'error': f'News API error: {str(e)}'}


def get_financial_news(query=None, page_size=10):
    try:
        if not NEWS_CONFIG['api_key']:
            return {'success': False, 'error': 'News API key not configured'}
        
        if not FINANCIAL_NEWS_CONFIG['enabled']:
            return {'success': False, 'error': 'Financial news is disabled'}
        
        url = f"{NEWS_CONFIG['base_url']}/everything"
        
        if query:
            search_query = f"{query} AND (stock OR market OR finance OR business OR economy)"
        else:
            search_query = "stock market OR finance OR business OR economy"
        
        params = {
            'q': search_query,
            'apiKey': NEWS_CONFIG['api_key'],
            'pageSize': min(page_size, NEWS_CONFIG['page_size']),
            'sortBy': 'publishedAt',
            'language': 'en'
        }
        
        if FINANCIAL_NEWS_CONFIG['sources']:
            params['domains'] = ','.join([f"{source}.com" for source in FINANCIAL_NEWS_CONFIG['sources']])
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            error_data = response.json()
            return {'success': False, 'error': error_data.get('message', 'Financial news API request failed')}
        
        data = response.json()
        
        if data['status'] != 'ok':
            return {'success': False, 'error': 'Failed to fetch financial news'}
        
        if not data.get('articles'):
            return {'success': False, 'error': 'No financial news articles found'}
        
        articles = []
        for article in data['articles'][:page_size]:
            articles.append({
                'title': article.get('title', 'No title'),
                'description': article.get('description', 'No description'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'author': article.get('author', 'Unknown'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'image_url': article.get('urlToImage', '')
            })
        
        return {
            'success': True,
            'data': {
                'articles': articles,
                'total_results': data.get('totalResults', len(articles)),
                'query': query,
                'type': 'financial'
            }
        }
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Financial news API timeout'}
    except Exception as e:
        return {'success': False, 'error': f'Financial news API error: {str(e)}'}


def detect_news_query(message):
    message_lower = message.lower()
    
    news_keywords = [
        'news', 'headline', 'headlines', 'latest news', 'breaking news',
        'current events', 'today news', 'recent news', 'news about',
        'whats happening', "what's happening", 'tell me about news',
        'any news', 'top news', 'trending news'
    ]
    
    financial_keywords = [
        'financial news', 'business news', 'market news', 'stock news',
        'economy news', 'finance news', 'trading news', 'wall street'
    ]
    
    is_news_query = any(keyword in message_lower for keyword in news_keywords)
    is_financial = any(keyword in message_lower for keyword in financial_keywords)
    
    if not is_news_query and not is_financial:
        return {'is_news': False}
    
    query = None
    for keyword in news_keywords + financial_keywords:
        if keyword in message_lower:
            parts = message_lower.split(keyword, 1)
            if len(parts) > 1:
                potential_query = parts[1].strip()
                for word in ['about', 'on', 'regarding', 'related to', '?', '.', '!']:
                    potential_query = potential_query.replace(word, '').strip()
                
                if potential_query:
                    query = potential_query
                    break
    
    return {
        'is_news': True,
        'is_financial': is_financial,
        'query': query
    }


def format_weather_for_ai(weather_data):
    if not weather_data['success']:
        return f"[Weather information unavailable: {weather_data['error']}]"
    
    data = weather_data['data']
    
    formatted = f"""
[Real-time Weather Data for {data['city']}, {data['country']}]:
- Current Temperature: {data['temperature']}°C (Feels like: {data['feels_like']}°C)
- Conditions: {data['description'].capitalize()} ({data['main_condition']})
- Temperature Range: {data['temp_min']}°C to {data['temp_max']}°C
- Humidity: {data['humidity']}%
- Wind: {data['wind_speed']} m/s
- Cloud Coverage: {data['clouds']}%
- Visibility: {data['visibility']}m
- Pressure: {data['pressure']} hPa
- Sunrise: {data['sunrise']} | Sunset: {data['sunset']}
- Last Updated: {data['timestamp']}

Please provide a natural, helpful, and conversational response based on this weather data.
"""
    return formatted


def format_wikipedia_for_ai(wiki_data):
    if not wiki_data['success']:
        return f"[Wikipedia information unavailable: {wiki_data['error']}]"
    
    data = wiki_data['data']
    
    formatted = f"""
[Wikipedia Information]:
Title: {data['title']}
Summary: {data['summary']}
Source: {data['url']}

Please provide a natural, conversational response based on this Wikipedia information.
"""
    return formatted


def format_news_for_ai(news_data):
    if not news_data['success']:
        return f"[News information unavailable: {news_data['error']}]"
    
    data = news_data['data']
    articles = data['articles']
    news_type = data.get('type', 'general')
    
    formatted = f"\n[Latest News Articles"
    if data.get('query'):
        formatted += f" about '{data['query']}'"
    formatted += f" - {news_type.capitalize()}]:\n\n"
    
    for i, article in enumerate(articles[:5], 1):
        formatted += f"{i}. {article['title']}\n"
        formatted += f"   Source: {article['source']}"
        
        if article['published_at']:
            try:
                pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                formatted += f" | {pub_date.strftime('%B %d, %Y')}"
            except:
                pass
        
        formatted += f"\n"
        
        if article['description'] and article['description'] != 'No description':
            formatted += f"   {article['description'][:200]}...\n"
        
        formatted += f"   URL: {article['url']}\n\n"
    
    formatted += f"Total articles found: {data['total_results']}\n\n"
    formatted += "Please provide a natural summary of these news articles.\n"
    
    return formatted


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Return available ElevenLabs voices"""
    try:
        # Try to fetch voices from ElevenLabs API
        headers = {
            'xi-api-key': ELEVENLABS_CONFIG['api_key']
        }
        response = requests.get(f"{ELEVENLABS_CONFIG['base_url']}/voices", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Organize voices by category
            organized_voices = {
                'female': [],
                'male': [],
                'other': []
            }
            
            for voice in data.get('voices', []):
                voice_info = {
                    'voice_id': voice['voice_id'],
                    'name': voice['name'],
                    'display_name': voice['name']
                }
                
                # Categorize by labels or use default
                labels = voice.get('labels', {})
                gender = labels.get('gender', '').lower()
                
                if gender == 'female':
                    organized_voices['female'].append(voice_info)
                elif gender == 'male':
                    organized_voices['male'].append(voice_info)
                else:
                    organized_voices['other'].append(voice_info)
            
            return jsonify(organized_voices)
        else:
            # Fallback to predefined voices
            return jsonify(ELEVENLABS_VOICES)
    except Exception as e:
        print(f"Error fetching ElevenLabs voices: {str(e)}")
        # Return predefined voices as fallback
        return jsonify(ELEVENLABS_VOICES)


@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Stream TTS audio directly from ElevenLabs to reduce latency"""
    try:
        data = request.json
        text = data.get('text', '')
        voice_id = data.get('voice_id', '21m00Tcm4TlvDq8ikWAM')

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        api_key = ELEVENLABS_CONFIG['api_key'].strip()
        if not api_key:
            return jsonify({'error': 'ElevenLabs API key not configured'}), 500

        # Use /stream endpoint + optimize_streaming_latency for lowest latency
        url = (
            f"{ELEVENLABS_CONFIG['base_url']}/text-to-speech/{voice_id}/stream"
            f"?optimize_streaming_latency=4&output_format=mp3_22050_32"
        )
        headers = {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
            'xi-api-key': api_key
        }
        payload = {
            'text': text,
            'model_id': 'eleven_turbo_v2_5',
            'voice_settings': {
                'stability': 0.5,
                'similarity_boost': 0.75,
                'style': 0.0,
                'use_speaker_boost': False  # disabled — adds latency
            }
        }

        print(f"[TTS] Streaming {len(text)} chars, voice={voice_id}")

        el_resp = requests.post(url, headers=headers, json=payload,
                                stream=True, timeout=30)

        if el_resp.status_code != 200:
            err = el_resp.text
            print(f"[TTS] ElevenLabs error {el_resp.status_code}: {err}")
            try:
                detail = el_resp.json().get('detail', {})
                err = detail.get('message', err) if isinstance(detail, dict) else detail
            except Exception:
                pass
            return jsonify({'error': f'ElevenLabs error ({el_resp.status_code}): {err}'}), el_resp.status_code

        def generate():
            for chunk in el_resp.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

        return Response(stream_with_context(generate()), content_type='audio/mpeg')

    except requests.exceptions.Timeout:
        return jsonify({'error': 'ElevenLabs timeout'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Unable to connect to ElevenLabs'}), 503
    except Exception as e:
        print(f"[TTS] Error: {str(e)}")
        return jsonify({'error': f'TTS error: {str(e)}'}), 500


@app.route('/api/stt', methods=['POST'])
def speech_to_text():
    """Convert speech to text using Azure Speech-to-Text"""
    webm_path = None
    wav_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file is required'}), 400

        audio_file = request.files['audio']

        # Save the uploaded webm file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            audio_file.save(tmp.name)
            webm_path = tmp.name

        # Convert webm → 16-kHz mono WAV (required by Azure STT REST API)
        wav_path = webm_path.replace('.webm', '.wav')
        subprocess.run(
            ['ffmpeg', '-y', '-i', webm_path, '-ar', '16000', '-ac', '1', wav_path],
            check=True, capture_output=True
        )

        # Call Azure STT REST API
        stt_url = (
            f"https://{AZURE_SPEECH_CONFIG['region']}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
        )
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_SPEECH_CONFIG['api_key'],
            'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
        }
        params = {'language': 'en-US', 'format': 'simple'}

        with open(wav_path, 'rb') as wav_file:
            response = requests.post(stt_url, headers=headers, params=params, data=wav_file, timeout=15)

        print(f"[Azure STT] Status: {response.status_code}")

        if response.status_code != 200:
            print(f"[Azure STT] Error: {response.text}")
            return jsonify({'error': f'Azure STT error ({response.status_code}): {response.text}'}), response.status_code

        result = response.json()
        recognition_status = result.get('RecognitionStatus', '')
        transcript = result.get('DisplayText', '').strip()

        print(f"[Azure STT] Status: {recognition_status} | Text: {transcript}")

        if recognition_status != 'Success' or not transcript:
            return jsonify({'error': f'Recognition failed: {recognition_status}'}), 422

        return jsonify({'success': True, 'text': transcript})

    except subprocess.CalledProcessError as e:
        print(f"[Azure STT] ffmpeg error: {e.stderr.decode()}")
        return jsonify({'error': 'Audio conversion failed'}), 500
    except Exception as e:
        print(f"[Azure STT] Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        for path in [webm_path, wav_path]:
            if path and os.path.exists(path):
                os.unlink(path)


def get_revinci_token():
    """Return a valid Keycloak bearer token, refreshing if expired."""
    now = time.time()
    # Refresh if token is missing or expires within 60 seconds
    if _revinci_token_cache['token'] and _revinci_token_cache['expires_at'] - now > 60:
        return _revinci_token_cache['token']

    token_url = (
        f"{REVINCI_CONFIG['keycloak_url']}"
        f"/realms/{REVINCI_CONFIG['keycloak_realm']}"
        f"/protocol/openid-connect/token"
    )
    data = {
        'grant_type': 'password',
        'client_id': REVINCI_CONFIG['keycloak_client_id'],
        'username': REVINCI_CONFIG['auth_username'],
        'password': REVINCI_CONFIG['auth_password'],
    }
    print(f"[Revinci] Fetching new token from {token_url}")
    print(f"[Revinci] Token request data: {data}")
    resp = requests.post(token_url, data=data, timeout=15, verify=REVINCI_SSL_VERIFY)
    print(f"[Revinci] Token response status: {resp.status_code}")
    print(f"[Revinci] Token response body: {resp.text}")
    resp.raise_for_status()
    token_data = resp.json()
    _revinci_token_cache['token'] = token_data['access_token']
    _revinci_token_cache['expires_at'] = now + token_data.get('expires_in', 300)
    print(f"[Revinci] Token refreshed, expires in {token_data.get('expires_in', 300)}s")
    return _revinci_token_cache['token']


def call_revinci_api(user_input, conversation_id=''):
    try:
        token = get_revinci_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {token}",
            'tenant-id': 'tessla'
        }
        payload = {
            'user_id': REVINCI_CONFIG['user_id'],
            'user_input': user_input,
            'conversation_id': conversation_id
        }
        print(f"[Revinci] POST {REVINCI_CONFIG['url']}")
        response = requests.post(REVINCI_CONFIG['url'], headers=headers, json=payload, timeout=30, verify=REVINCI_SSL_VERIFY)
        print(f"[Revinci] Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'content': data.get('content', ''),
                'conversation_id': data.get('conversation_id', '')
            }
        else:
            print(f"[Revinci] Error {response.status_code}: {response.text}")
            return {'success': False, 'error': f"Status {response.status_code}"}
    except Exception as e:
        print(f"[Revinci] Exception: {str(e)}")
        return {'success': False, 'error': str(e)}


@app.route('/api/chat', methods=['POST'])
@app.route('/api/opportunity', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message') or data.get('user_input', '')
        session_id = data.get('session_id') or data.get('conversation_id', 'default')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        revinci_conv_id = revinci_conversation_ids.get(session_id, '')
        revinci_result = call_revinci_api(message, revinci_conv_id)

        if revinci_result['success']:
            assistant_message = revinci_result['content']
            revinci_conversation_ids[session_id] = revinci_result['conversation_id']
        else:
            print(f"Revinci API error: {revinci_result.get('error')}")
            return jsonify({'error': 'Failed to get a response. Please try again.'}), 502

        return jsonify({
            'response': assistant_message,
            'session_id': session_id
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/weather', methods=['GET'])
def get_weather():
    try:
        city = request.args.get('city')
        
        if not city:
            return jsonify({'error': 'City parameter is required'}), 400
        
        weather_data = get_weather_data(city)
        
        if weather_data['success']:
            return jsonify(weather_data['data'])
        else:
            return jsonify({'error': weather_data['error']}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        city = request.args.get('city')
        days = int(request.args.get('days', 5))
        
        if not city:
            return jsonify({'error': 'City parameter is required'}), 400
        
        if days < 1 or days > 5:
            return jsonify({'error': 'Days must be between 1 and 5'}), 400
        
        forecast_data = get_weather_forecast(city, days)
        
        if forecast_data['success']:
            return jsonify(forecast_data)
        else:
            return jsonify({'error': forecast_data['error']}), 404
            
    except ValueError:
        return jsonify({'error': 'Days parameter must be a number'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/wikipedia', methods=['GET'])
def get_wikipedia():
    try:
        query = request.args.get('query')
        sentences = int(request.args.get('sentences', 3))
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        if sentences < 1 or sentences > 10:
            return jsonify({'error': 'Sentences must be between 1 and 10'}), 400
        
        wiki_data = search_wikipedia(query, sentences)
        
        if wiki_data['success']:
            return jsonify(wiki_data['data'])
        else:
            return jsonify({'error': wiki_data['error']}), 404
            
    except ValueError:
        return jsonify({'error': 'Sentences parameter must be a number'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/news', methods=['GET'])
def get_news():
    try:
        query = request.args.get('query')
        category = request.args.get('category')
        country = request.args.get('country', 'us')
        page_size = int(request.args.get('page_size', 10))
        
        if page_size < 1 or page_size > 100:
            return jsonify({'error': 'Page size must be between 1 and 100'}), 400
        
        news_data = get_general_news(query=query, category=category, country=country, page_size=page_size)
        
        if news_data['success']:
            return jsonify(news_data['data'])
        else:
            return jsonify({'error': news_data['error']}), 404
            
    except ValueError:
        return jsonify({'error': 'Page size must be a number'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/financial-news', methods=['GET'])
def get_financial_news_endpoint():
    try:
        query = request.args.get('query')
        page_size = int(request.args.get('page_size', 10))
        
        if page_size < 1 or page_size > 100:
            return jsonify({'error': 'Page size must be between 1 and 100'}), 400
        
        news_data = get_financial_news(query=query, page_size=page_size)
        
        if news_data['success']:
            return jsonify(news_data['data'])
        else:
            return jsonify({'error': news_data['error']}), 404
            
    except ValueError:
        return jsonify({'error': 'Page size must be a number'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in conversation_sessions:
            conversation_sessions[session_id] = [
                {'role': 'system', 'content': SYSTEM_PROMPT}
            ]
        
        return jsonify({'message': 'Conversation cleared successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    weather_api_configured = WEATHER_CONFIG['api_key'] != 'YOUR_OPENWEATHERMAP_API_KEY'
    news_api_configured = NEWS_CONFIG['api_key'] != 'YOUR_NEWSAPI_KEY'
    
    return jsonify({
        'status': 'healthy',
        'service': 'Conversational AI Assistant',
        'azure_openai': 'configured',
        'weather_api': 'configured' if weather_api_configured else 'not configured',
        'wikipedia_api': 'available',
        'news_api': 'configured' if news_api_configured else 'not configured',
        'azure_stt': 'configured' if AZURE_SPEECH_CONFIG['api_key'] else 'not configured',
        'elevenlabs_tts': 'configured',
        'active_sessions': len(conversation_sessions)
    })


if __name__ == '__main__':
    import os
    
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 Conversational AI Assistant")
    print("   Weather | Wikipedia | News | Azure STT | ElevenLabs TTS")
    print("="*70)
    
    print("✅ Azure OpenAI: Configured")
    
    if not WEATHER_CONFIG['api_key']:
        print("⚠️  Weather API: Not Configured")
        print("   → Sign up at https://openweathermap.org/api")
    else:
        print("✅ Weather API: Configured")
    
    print("✅ Wikipedia API: Available (no key needed)")
    
    if not NEWS_CONFIG['api_key']:
        print("⚠️  News API: Not Configured")
        print("   → Sign up at https://newsapi.org")
    else:
        print("✅ News API: Configured")

    if not AZURE_SPEECH_CONFIG['api_key']:
        print("⚠️  Azure STT: Not Configured")
    else:
        print("✅ Azure STT: Configured")
    print("✅ ElevenLabs TTS: Configured")
    
    print("\n🌐 Server: https://localhost:5001")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=True, ssl_context='adhoc')

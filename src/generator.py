import os
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch

# Get the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define folder paths relative to the project root
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "real")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "fake")

# Check if the input folder exists (required for source audio files)
if not os.path.exists(INPUT_FOLDER):
    print(f"Error: Input folder not found: {INPUT_FOLDER}")
    print("Please add the real voice files to the 'data/real' folder.")
    exit()

# Create the output folder (data/fake) if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Info: {OUTPUT_FOLDER} folder has been created.")

# --- MODEL AND HARDWARE SETTINGS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# --- TEXT DATA ---
texts = [
    "Teknolojinin hızla gelişmesiyle birlikte yapay zekâ sistemleri artık insan sesini gerçeğinden ayırt edilemeyecek seviyede taklit edebiliyor.",
    "Bugün öğleden sonra kütüphaneye gidip sinyal işleme projem için gerekli olan temel makine öğrenimi kaynaklarını detaylıca araştıracağım.",
    "Gelecekte siber güvenlik uzmanlarının en büyük uğraşlarından biri, internet ortamındaki derin sahte içerikleri tespit etmek olacaktır.",
    "Eski zamanlarda insanlar haberleşmek için posta güvercinlerini kullanırken, günümüzde saniyeler içinde dünyanın öbür ucuna veri gönderebiliyoruz.",
    "Bahçedeki ağaçların yaprakları sonbaharın gelmesiyle birlikte sararmaya ve hafif rüzgârın etkisiyle yerlere dökülmeye başladı.",
    "Üniversite yıllarım boyunca edindiğim en değerli tecrübe, karmaşık problemleri takım çalışmasıyla nasıl çözeceğimi öğrenmek oldu.",
    "Güneşli bir pazar sabahı erkenden uyanıp sahilde uzun bir yürüyüş yapmak insana müthiş bir enerji ve huzur veriyor.",
    "Bu akşamki akşam yemeği menüsünde taze sebzelerle hazırlanmış güzel bir zeytinyağlı yemek ve yanında ev yapımı yoğurt bulunuyor.",
    "Bilim kurgu filmlerinde gördüğümüz otonom araçlar ve akıllı şehir sistemleri artık günlük hayatımızın bir parçası haline gelmeye başladı.",
    "Müziğin insan ruhu üzerindeki iyileştirici etkisini anlamak için notaların frekanslarındaki o muazzam uyuma odaklanmak yeterlidir.",
    "Laboratuvar çalışmaları sırasında topladığımız verilerin doğruluğunu teyit etmek için deneyleri birkaç kez tekrarlamamız gerekebilir.",
    "Kitap okumak, insanın hayal gücünü geliştirmesinin yanı sıra olaylara farklı pencerelerden bakabilme yeteneği kazanmasını sağlar.",
    "Şehrin kalabalık ve gürültülü caddelerinde yürürken bazen doğanın o sessiz ve sakin atmosferini ne kadar özlediğimi fark ediyorum.",
    "Küresel iklim değişikliğiyle mücadele etmek için fosil yakıt kullanımını azaltmalı ve yenilenebilir enerji kaynaklarına yönelmeliyiz.",
    "Bir dili tam anlamıyla öğrenebilmek için sadece kelime ezberlemek yetmez, o dilin kültürünü ve günlük kullanım kalıplarını da bilmek gerekir.",
    "Bilgisayar oyunları endüstrisi, son yıllarda elde ettiği devasa gelirlerle sinema ve müzik sektörünü geride bırakmayı başardı.",
    "Sabahları içilen bir fincan sıcak kahve, insanın güne daha odaklanmış ve zinde bir şekilde başlamasına yardımcı olan önemli bir ritüeldir.",
    "Uzay araştırmaları sayesinde evrenin sırlarını yavaş yavaş keşfederken, dünya dışı yaşam ihtimali üzerine yeni teoriler üretiyoruz.",
    "Sanatçılar, eserlerini oluştururken sadece estetik kaygılarla değil, toplumsal sorunlara ayna tutma amacıyla da hareket ederler.",
    "Başarılı bir kariyer inşa etmek için sürekli öğrenmeye açık olmak, disiplinli çalışmak ve karşılaşılan zorluklar karşısında pes etmemek gerekir.",
    "Teknolojinin hızla gelişmesiyle birlikte yapay zekâ sistemleri artık insan sesini gerçeğinden ayırt edilemeyecek seviyede taklit edebiliyor.",
    "Bugün öğleden sonra kütüphaneye gidip sinyal işleme projem için gerekli olan temel makine öğrenimi kaynaklarını detaylıca araştıracağım.",
    "Gelecekte siber güvenlik uzmanlarının en büyük uğraşlarından biri, internet ortamındaki derin sahte içerikleri tespit etmek olacaktır.",
    "Eski zamanlarda insanlar haberleşmek için posta güvercinlerini kullanırken, günümüzde saniyeler içinde dünyanın öbür ucuna veri gönderebiliyoruz.",
    "Bahçedeki ağaçların yaprakları sonbaharın gelmesiyle birlikte sararmaya ve hafif rüzgârın etkisiyle yerlere dökülmeye başladı.",
    "Üniversite yıllarım boyunca edindiğim en değerli tecrübe, karmaşık problemleri takım çalışmasıyla nasıl çözeceğimi öğrenmek oldu.",
    "Güneşli bir pazar sabahı erkenden uyanıp sahilde uzun bir yürüyüş yapmak insana müthiş bir enerji ve huzur veriyor.",
    "Bu akşamki akşam yemeği menüsünde taze sebzelerle hazırlanmış güzel bir zeytinyağlı yemek ve yanında ev yapımı yoğurt bulunuyor.",
    "Bilim kurgu filmlerinde gördüğümüz otonom araçlar ve akıllı şehir sistemleri artık günlük hayatımızın bir parçası haline gelmeye başladı.",
    "Müziğin insan ruhu üzerindeki iyileştirici etkisini anlamak için notaların frekanslarındaki o muazzam uyuma odaklanmak yeterlidir.",
    "Laboratuvar çalışmaları sırasında topladığımız verilerin doğruluğunu teyit etmek için deneyleri birkaç kez tekrarlamamız gerekebilir.",
    "Kitap okumak, insanın hayal gücünü geliştirmesinin yanı sıra olaylara farklı pencerelerden bakabilme yeteneği kazanmasını sağlar.",
    "Şehrin kalabalık ve gürültülü caddelerinde yürürken bazen doğanın o sessiz ve sakin atmosferini ne kadar özlediğimi fark ediyorum.",
    "Küresel iklim değişikliğiyle mücadele etmek için fosil yakıt kullanımını azaltmalı ve yenilenebilir enerji kaynaklarına yönelmeliyiz.",
    "Bir dili tam anlamıyla öğrenebilmek için sadece kelime ezberlemek yetmez, o dilin kültürünü ve günlük kullanım kalıplarını da bilmek gerekir.",
    "Bilgisayar oyunları endüstrisi, son yıllarda elde ettiği devasa gelirlerle sinema ve müzik sektörünü geride bırakmayı başardı.",
    "Sabahları içilen bir fincan sıcak kahve, insanın güne daha odaklanmış ve zinde bir şekilde başlamasına yardımcı olan önemli bir ritüeldir.",
    "Uzay araştırmaları sayesinde evrenin sırlarını yavaş yavaş keşfederken, dünya dışı yaşam ihtimali üzerine yeni teoriler üretiyoruz.",
    "Sanatçılar, eserlerini oluştururken sadece estetik kaygılarla değil, toplumsal sorunlara ayna tutma amacıyla da hareket ederler.",
    "Başarılı bir kariyer inşa etmek için sürekli öğrenmeye açık olmak, disiplinli çalışmak ve karşılaşılan zorluklar karşısında pes etmemek gerekir."
]

# --- PROCESSING LOOP ---
audio_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.wav')])

for i, file_name in enumerate(audio_files):
    if i >= len(texts):
        print("Warning: More audio files found than provided texts. Processing stopped.")
        break

    try:
        input_path = os.path.join(INPUT_FOLDER, file_name)
        
        # Generate output file name with 'fake_' prefix
        output_name = "fake_" + file_name.replace("real_", "")
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        
        # Generation process
        wav = multilingual_model.generate(texts[i], audio_prompt_path=input_path, language_id="tr")
        ta.save(output_path, wav, multilingual_model.sr)
        
        print(f"[{i+1}/{len(audio_files)}] Generated: {output_name}")
        
    except Exception as e:
        print(f"Error occurred during processing ({file_name}): {e}")

print("\nProcessing complete. Generated audio files saved in 'data/fake' directory.")
#5010890
from spellchecker import SpellChecker
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Initialisierung des Spellcheckers für Deutsch
spell = SpellChecker(language='de')

# Laden der benutzerdefinierten Wörter
with open('/data/custom_dictionary.txt') as f:
    custom_words = f.read().splitlines()
spell.word_frequency.load_words(custom_words)

# Statistik-Dictionaries
correction_count = {}
original_to_corrected = {}

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :load_data.max_len]

    output_text = []
    total_words_checked = 0
    total_corrections = 0

    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(tokenizer.num_to_char(res)).numpy().decode("utf-8")

        #Spellchecker
        corrected_text = []
        for word in res.split():
            total_words_checked += 1
            corrected = spell.correction(word)
            if corrected != word:
                total_corrections += 1
                correction_count[word] = correction_count.get(word, 0) + 1
                original_to_corrected[word] = corrected
            corrected_text.append(corrected)
        
        output_text.append(' '.join(corrected_text))
    # Ergebnisse zurückgeben
    return output_text, total_words_checked, total_corrections, correction_count, original_to_corrected


# Statistiken anzeigen
print("Gesamtzahl der überprüften Wörter:", total_words)
print("Gesamtzahl der Korrekturen:", total_corrections)
print("Korrekturhäufigkeit:", correction_count)
print("Original zu korrigierten Wörtern:", original_to_corrected)

# Statistiken in eine Log-Datei schreiben
log_file_path = 'prediction_log.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write("Gesamtzahl der überprüften Wörter: {}\n".format(total_words_checked))
    log_file.write("Gesamtzahl der Korrekturen: {}\n".format(total_corrections))
    log_file.write("Korrekturhäufigkeit: {}\n".format(correction_count))
    log_file.write("Original zu korrigierten Wörtern: {}\n".format(original_to_corrected))

print(f"Statistiken wurden in '{log_file_path}' gespeichert.")

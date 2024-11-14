function main(){
  set -ex
  speech_ci_path=`pwd`

  echo "Start asr"
  cd ${speech_ci_path}/asr
  bash deepspeech2_online_model_test.sh
  python error_rate_test.py
  python mask_test.py
  python reverse_pad_list.py
  echo "End asr"

  echo "Start TTS"
  cd ${speech_ci_path}/tts
  python test_data_table.py
  python test_enfrontend.py
  python test_mixfrontend.py
  echo "End TTS"

  echo "Start Vector"
  cd ${speech_ci_path}/vector
  python test_augment.py
  echo "End Vector"

  echo "Start cli"
  cd ${speech_ci_path}/cli
  bash test_cli.sh
  echo "End cli"

  echo "Start server"
  cd ${speech_ci_path}/server/offline
  bash test_server_client.sh
  echo "End server"
}

main

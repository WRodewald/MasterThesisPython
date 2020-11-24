
id = 1

items = ['f1_a',\
         'f1_i',\
         'f1_o',\
         'f4_a',\
         'f4_i',\
         'f4_o',\
         'f5_a',\
         'f5_i',\
         'f5_o',\
         'f6_a',\
         'f6_i',\
         'f6_o',\
         'f8_a',\
         'f8_i',\
         'f8_o',\
         'm1_a',\
         'm1_i',\
         'm1_o',\
         'm2_a',\
         'm2_i',\
         'm2_o',\
         'm4_a',\
         'm4_i',\
         'm4_o',\
         'm5_a',\
         'm5_i',\
         'm5_o',\
         'm8_a',\
         'm8_i',\
         'm8_o']


for i in items:

    print("\n\n\
    - type: mushra\n\
      id: {0}\n\
      name: Mono Trial\n\
      content: Please rate the quality of all sounds in relation to the reference.\n\
      showmp3eform: true\n\
      enableLooping: true \n\
      randomize: true \n\
      reference: configs/resources/audio/results_mp3/{0}/audio/original.mp3\n\
      createAnchor35: false\n\
      createAnchor70: false\n\
      stimuli:\n\
          C1: configs/resources/audio/results_mp3/{0}/audio/harmonic.mp3\n\
          C2: configs/resources/audio/results_mp3/{0}/audio/synthesis_estimated.mp3\n\
          C3: configs/resources/audio/results_mp3/{0}/audio/synthesized_{0}.mp3\n\
          C4: configs/resources/audio/results_mp3/{0}/audio/anchor35.mp3".format(i))
#         C5: configs/resources/audio/results_mp3/{0}/audio/anchor70.mp3\n".format(i))



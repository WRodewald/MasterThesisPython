
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
    - type: quality_ab\n\
      id: quality_{0}\n\
      name: Trial $PAGE of $NUM_PAGES\n\
      content: Please answer the questions bellow.\n\
      showmp3eform: true\n\
      enableLooping: true \n\
      reference: configs/resources/audio/results_mp3/{0}/audio/original.mp3\n\
      stimulus:  configs/resources/audio/results_mp3/{0}/audio/synthesized_{0}.mp3\n\
      reversed_order: false\n\
      scale:\n\
        - Value: -4\n\
          Label: 4<br><small>(strongly)</small>\n\
        - Value: -3\n\
          Label: 3<br>&nbsp;\n\
        - Value: -2\n\
          Label: 2<br>&nbsp;\n\
        - Value: -1\n\
          Label: 1<br>&nbsp;\n\
        - Value: 0\n\
          Label: 0<br><small>(neutral)</small>\n\
        - Value: 1\n\
          Label: 1<br>&nbsp;\n\
        - Value: 2\n\
          Label: 2<br>&nbsp;\n\
        - Value: 3\n\
          Label: 3<br>&nbsp;\n\
        - Value: 4\n\
          Label: 4<br><small>(strongly)</small>\n\
      qualityItems:\n\
          - Item: Breathiness\n\
            Label: Breathiness\n\
            Description: Which of the two samples sounds more breathy?\n\
          - Item: Roughness\n\
            Label: Roughness\n\
            Description: Which of the two samples sounds more rough?\n\
          - Item: Brightness\n\
            Label: Brightness\n\
            Description: Which of the two samples sounds brighter?\n\
          - Item: Naturalness\n\
            Label: Naturalness\n\
            Description: Which of the two samples sounds more natural?".format(i))

    if(False):
      print("\n\n\
      - type: quality_ab\n\
        id: quality_{0}_rev\n\
        name: Trial $PAGE of $NUM_PAGES\n\
        content: Please answer the questions bellow.\n\
        showmp3eform: true\n\
        enableLooping: true \n\
        reference: configs/resources/audio/results_mp3/{0}/audio/original.mp3\n\
        stimulus:  configs/resources/audio/results_mp3/{0}/audio/synthesized_{0}.mp3\n\
        reversed_order: true\n\
        scale:\n\
          - Value: -4\n\
            Label: 4<br><small>(strongly)</small>\n\
          - Value: -3\n\
            Label: 3<br>&nbsp;\n\
          - Value: -2\n\
            Label: 2<br>&nbsp;\n\
          - Value: -1\n\
            Label: 1<br>&nbsp;\n\
          - Value: 0\n\
            Label: 0<br><small>(neutral)</small>\n\
          - Value: 1\n\
            Label: 1<br>&nbsp;\n\
          - Value: 2\n\
            Label: 2<br>&nbsp;\n\
          - Value: 3\n\
            Label: 3<br>&nbsp;\n\
          - Value: 4\n\
            Label: 4<br><small>(strongly)</small>\n\
        qualityItems:\n\
            - Item: Breathiness\n\
              Label: Breathiness\n\
              Description: Which of the two samples sounds more breathy?\n\
            - Item: Roughness\n\
              Label: Roughness\n\
              Description: Which of the two samples sounds more rough?\n\
            - Item: Brightness\n\
              Label: Brightness\n\
              Description: Which of the two samples sounds brighter?\n\
            - Item: Naturalness\n\
              Label: Naturalness\n\
              Description: Which of the two samples sounds more natural?".format(i))


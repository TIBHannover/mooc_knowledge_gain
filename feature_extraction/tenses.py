

class Tenses:
    """ Class that calculate time tense specific features
    """
    def __init__(self):
        """ Initialise class-object with default values
        """
        # Present
        self.sim_pres = 0  # Simple Present
        self.pres_prog = 0  # Present Progressive (includes going-to future because it is very similar)
        self.pres_perf = 0  # Present Perfect
        self.pres_perf_prog = 0  # Present Perfect Progressive

        # Past
        self.sim_pas = 0  # Simple Past
        self.pas_prog = 0  # Past Progressive
        self.pas_perf = 0  # Past Perfect
        self.pas_perf_prog = 0  # Past Perfect Progressive

        # Future
        self.will = 0  # Will Future
        self.fu_prog = 0  # Future Progressive
        self.fu_perf = 0  # Future Perfect
        self.fu_perf_prog = 0  # Future Perfect Progressive

        # Conditional
        self.cond_sim = 0  # Conditional Simple
        self.cond_prog = 0  # Conditonal Progressive
        self.cond_perf = 0  # Condtional Perfect
        self.cond_perf_prog = 0  # Conditonal Perfect Progressive

        # Participle
        self.gerund = 0  # Gerund , Present Participle
        self.perf_part = 0  # Perfect Participle

        # Infinitive
        self.inf = 0  # Present Infinitive
        self.perf_inf = 0  # perfect Infinitive

        # tense version
        self.active = 0
        self.passive = 0
        
        # help variable
        self.is_passive = False  # shows if current tense is passive or not
        self.was_prog = False  # Shows if previous was progressive
        self.was_perf = False  # Shows if previous was perfect

    def get_features(self):
        """ Generates array of the time tenses
        :return: Time tenses as array
        """
        amount = self.active + self.passive
        if not amount:
            amount = 1
        features = []
        for element in self.__dict__.items():
            if element[0] not in ["is_passive", "was_prog", "was_perf"]:
                features.append(round(element[1], 6))  # no ratio
                features.append(round(element[1] / amount, 6))  # ratio
        return features

    def print_tenses(self):
        """ For tests to see if verb tenses are correct
        """
        print("Active:" + str(self.active))
        print("Passive:" + str(self.passive))
        print()
        print("Present:")
        print("Simple Present: " + str(self.sim_pres))
        print("Present Progressiv: " + str(self.pres_prog))
        print("Present Perfect: " + str(self.pres_perf))
        print("Present Perfect Progressive: " + str(self.pres_perf_prog))
        print()
        print("Past:")
        print("Simple Past: " + str(self.sim_pas))
        print("Past Progressiv: " + str(self.pas_prog))
        print("Past Perfect: " + str(self.pas_perf))
        print("Past Perfect Progressive: " + str(self.pas_perf_prog))
        print()
        print("Future:")
        print("Will-Future: " + str(self.will))
        print("Future Progressive: " + str(self.fu_prog))
        print("Future Perfect: " + str(self.fu_perf))
        print("Future Perfect Progressive: " + str(self.fu_perf_prog))
        print()
        print("Conditional:")
        print("Conditional Simple: " + str(self.cond_sim))
        print("Conditional Progressive: " + str(self.cond_prog))
        print("Conditional Perfect: " + str(self.cond_perf))
        print("Conditional Perfect Progressive: " + str(self.cond_perf_prog))
        print()
        print("Infinitive:")
        print("Present Infinitive: " + str(self.inf))
        print("Perfect Infinitive: " + str(self.perf_inf))
        print()
        print("Participle: ")
        print("Gerund/Present Participle: " + str(self.gerund))
        print("Perfect Participle: " + str(self.perf_part))
        print()

    def __get_time(self, time, token, pos):
        """ Checks which time tense it could be and set them to previous token, previous pos for the next token
        :param time: array of booleans to set the current time
        :param token: current token to analyze
        :param pos: the xpos-tag of the token
        :return: modified time-array, previous token and previous pos tag
        """
        prev_token = ''
        prev_pos = ''
        if pos == 'MD':
            if token == 'will' or token == "shall":
                time[2] = True  # Future
                prev_token, prev_pos = 'will/shall', pos
            elif token == 'would':
                time[3] = True  # Conditional
                prev_token, prev_pos = 'would', pos
            else:
                pass
        elif pos == 'TO':
            time[4] = True  # Infinitve
            prev_token, prev_pos = token, pos
        elif pos in ["VBZ", "VBP"]:
            time[0] = True  # Present
            prev_token, prev_pos = token, pos
        elif pos == "VB":
            self.sim_pres += 1
            self.active += 1
        elif pos == "VBD":
            time[1] = True  # Past
            prev_token, prev_pos = token, pos
        elif pos == "VBN":  # for example formalized model (formalized is here like an adjective)
            pass
        elif pos == "VBG":  # Participle
            time[5] = True
            prev_token, prev_pos = token, pos
        return time, prev_token, prev_pos

    def __get_present(self, time, prev_token, prev_pos, token, pos):
        """ The present can have different forms. This function checks which tense could be possible and if one tense
        is recognized, it will be incremented. At the end of the function the current token and current pos tag (xpos)
         are set to the previous ones. Possible tenses are:
        Simple Present ->  Active: VB/VBZ/VBP; do/does + VB; Passive: am/is/are + VBN
        Present Progressive -> Active: am/is/are + VBG; Passive: is + being + VBN
        Present Perfect -> Active: has/have + VBN; Passive: has/have + been + VBN
        Present Perfect Progressive -> Active: has/have + been + VBG
        :param time: time-array that contains boolean. The current tense has the value True (here present)
        :param prev_token: the previous token to identify the possible tenses
        :param prev_pos: the previous pos tag (xpos) to identify the possible tenses
        :param token: the current token
        :param pos: the current pos tag (xpos)
        :return: time-array, the previous token and the previous pos tag (xpos)
        """
        if prev_token in ["do", "does"] and pos == "VB":
            prev_token, prev_pos = token, pos
        elif (prev_token in ["is", "am", "are"] or prev_pos == "VBN") and pos == "VBG":
            prev_token, prev_pos = token, pos
        elif prev_token in ["is", "am", "are"] and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_token in ["has", "have"] and pos == "VBN":
            self.was_perf = True
            prev_token, prev_pos = token, pos
        elif prev_token == "been" and self.was_perf and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_pos == "VBG" and pos == "VBN":
            self.is_passive = True
            self.was_prog = True
            prev_token, prev_pos = token, pos
        else:
            if prev_pos == "VBN":
                if self.is_passive:
                    if self.was_prog:
                        self.pres_prog += 1
                    elif self.was_perf:
                        self.pres_perf += 1
                    else:
                        self.sim_pres += 1
                    self.passive += 1
                    self.is_passive = False
                    self.was_prog = False
                    self.was_perf = False
                else:
                    self.pres_perf += 1
                    self.active += 1
                time[0] = False  # set present to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            elif prev_pos == "VBG":
                if self.was_perf:
                    self.pres_perf_prog += 1
                    self.active += 1
                else:
                    self.pres_prog += 1
                    self.active += 1
                self.was_perf = False
                time[0] = False  # set present to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            else:
                self.sim_pres += 1
                self.active += 1
                time[0] = False  # set present to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
        return time, prev_token, prev_pos

    def __get_past(self, time, prev_token, prev_pos, token, pos):
        """ The past can have different forms. This function checks which tense could be possible and if one tense
        is recognized, it will be incremented. At the end of the function the current token and current pos tag (xpos)
         are set to the previous ones. Possible tenses are:
         Simple Past -> Active: VBD; did + VB; Passive:  was/were + VBN
        Past Progressive -> Active: was/were + VBG; Passive: was/were + being + VBN
        Past Perfect -> Active: had + VBN; Passive: had + been + VBN
        Past Perfect Progressive -> Active: had + been + VBG;
        :param time: time-array that contains boolean. The current tense has the value True (here past)
        :param prev_token: the previous token to identify the possible tenses
        :param prev_pos: the previous pos tag (xpos) to identify the possible tenses
        :param token: the current token
        :param pos: the current pos tag (xpos)
        :return: time-array, the previous token and the previous pos tag (xpos)
        """
        if prev_token == "did" and pos == "VB":
            prev_token, prev_pos = token, pos
        elif (prev_token in ["was", "were"] or prev_pos == "VBN") and pos == "VBG":
            prev_token, prev_pos = token, pos
        elif prev_token in ["was", "were"] and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_token == "had" and pos == "VBN":
            self.was_perf = True
            prev_token, prev_pos = token, pos
        elif prev_token == "been" and self.was_perf and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_pos == "VBG" and pos == "VBN":
            self.is_passive = True
            self.was_prog = True
            prev_token, prev_pos = token, pos
        else:
            if prev_pos == "VBN":
                if self.is_passive:
                    if self.was_prog:
                        self.pas_prog += 1
                    elif self.was_perf:
                        self.pas_perf += 1
                    else:
                        self.sim_pas += 1
                    self.passive += 1
                    self.was_prog = False
                    self.is_passive = False
                else:
                    self.pas_perf += 1
                    self.active += 1
                time[1] = False  # set past to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            elif prev_pos == "VBG":
                if self.was_perf:
                    self.pas_perf_prog += 1
                    self.active += 1
                else:
                    self.pas_prog += 1
                    self.active += 1
                self.was_perf = False
                time[1] = False  # set past to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            else:
                self.sim_pas += 1
                self.active += 1
                time[1] = False  # set past to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
        return time, prev_token, prev_pos

    def __get_future(self, time, prev_token, prev_pos, token, pos):
        """ The future can have different forms. This function checks which tense could be possible and if one tense
        is recognized, it will be incremented. At the end of the function the current token and current pos tag (xpos)
         are set to the previous ones. Possible tenses are:
        Will Future -> Active: will/shall + VB; Passive: will/shall + be + VBN
        Future Progressive -> Active: will/shall + be + VBG
        Future Perfect -> Active: will/shall + have + VBN
        Future Perfect Progressive -> Active: will/shall + have + been + VBG
        Conditional Simple -> would + VB; Passive: would + be + VBN
        :param time: time-array that contains boolean. The current tense has the value True (here future)
        :param prev_token: the previous token to identify the possible tenses
        :param prev_pos: the previous pos tag (xpos) to identify the possible tenses
        :param token: the current token
        :param pos: the current pos tag (xpos)
        :return: time-array, the previous token and the previous pos tag (xpos)
        """
        if prev_token == "will/shall" and pos == "VB":
            prev_token, prev_pos = token, pos
        elif prev_token == "be" and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_token == "be" and pos == "VBG":
            self.was_prog = True
            prev_token, prev_pos = token, pos
        elif prev_token == "have" and pos == "VBN":
            prev_token, prev_pos = token, pos
        elif prev_token == "been" and pos == "VBG":
            prev_token, prev_pos = token, pos
        else:
            if prev_pos == "VB":
                self.will += 1
                self.active += 1
                time[2] = False  # set future to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            elif prev_pos == "VBN":
                if self.is_passive:
                    self.will += 1
                    self.passive += 1
                else:
                    self.fu_perf += 1
                    self.active += 1
                time[2] = False  # set future to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            elif prev_pos == "VBG":
                if self.was_prog:
                    self.was_prog = False
                    self.fu_prog += 1
                    self.active += 1
                else:
                    self.fu_perf_prog += 1
                    self.active += 1
                time[2] = False  # set future to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            else:
                self.will += 1
                self.active += 1
                time[2] = False  # set future to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
        return time, prev_token, prev_pos

    def __get_conditional(self, time, prev_token, prev_pos, token, pos):
        """ The conditional can have different forms. The function checks which tense could be possible and if one tense
        is recognized, it will be incremented. At the end of the function the current token and current pos tag (xpos)
         are set to the previous ones. Possible tenses are:
        Conditional Simple -> Active: would + VB; Passive: would + be + VBN
        Conditional Progressive -> Active: would + be + VBG
        Conditional Perfect -> Active: would + have + VBN; Passive: would + have + been + VBN
        Condtional Perfect Progressive -> Active: would + have + been + VBG
        :param time: time-array that contains boolean. The current tense has the value True (here conditional)
        :param prev_token: the previous token to identify the possible tenses
        :param prev_pos: the previous pos tag (xpos) to identify the possible tenses
        :param token: the current token
        :param pos: the current pos tag (xpos)
        :return: time-array, the previous token and the previous pos tag (xpos)
        """
        if prev_token == "would" and pos == "VB":
            prev_token, prev_pos = token, pos
        elif prev_token == "be" and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_token == "been" and self.was_perf and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_token in ["be", "been"] and pos == "VBG":
            prev_token, prev_pos = token, pos
        elif prev_token == "have" and pos == "VBN":
            self.was_perf = True
            prev_token, prev_pos = token, pos
        else:
            if prev_pos == "VBN":
                if self.is_passive:
                    if self.was_perf:
                        self.cond_perf += 1
                    else:
                        self.cond_sim += 1
                    self.passive += 1
                    self.is_passive = False
                elif self.was_perf:
                    self.cond_perf += 1
                    self.active += 1
                    self.was_perf = False
                else:
                    self.cond_prog += 1
                    self.active += 1
                time[3] = False  # set conditional to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            elif prev_pos == "VBG":
                if self.was_perf:
                    self.cond_perf_prog += 1
                    self.active += 1
                    self.was_perf = False
                else:
                    self.cond_prog += 1
                    self.active += 1
                time[3] = False  # set conditional to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            else:
                self.cond_sim += 1
                self.active += 1
                time[3] = False  # set conditional to False to check the next tense
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
        return time, prev_token, prev_pos

    def __get_infinitive(self, time, prev_token, prev_pos, token, pos):
        """ The infinitive can have different forms. The function checks which tense could be possible and if one tense
        is recognized, it will be incremented. At the end of the function the current token and current pos tag (xpos)
         are set to the previous ones. Possible tenses are:
        Present Infinitive -> Active: to + VB; Passive: to + be + VBN
        Perfect Infinitive -> Active: to + have + VBN; Passive: to + have + been + VBN
        :param time: time-array that contains boolean. The current tense has the value True (here infinitive)
        :param prev_token: the previous token to identify the possible tenses
        :param prev_pos: the previous pos tag (xpos) to identify the possible tenses
        :param token: the current token
        :param pos: the current pos tag (xpos)
        :return: time-array, the previous token and the previous pos tag (xpos)
        """
        if prev_pos == "TO" and pos == "VB":
            prev_token, prev_pos = token, pos
        elif prev_token == "be" and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_token == "been" and pos == "VBN":
            self.is_passive = True
            self.was_perf = True
            prev_token, prev_pos = token, pos
        elif prev_token == "have" and pos == "VBN":
            prev_token, prev_pos = token, pos
        else:
            if prev_pos == "VBN":
                if self.is_passive:
                    if self.was_perf:
                        self.perf_inf += 1
                        self.passive += 1
                    else:
                        self.inf += 1
                        self.passive += 1
                else:
                    self.perf_inf += 1
                    self.active += 1
                time[4] = False  # set infinitive to False to check the next tense
            elif prev_pos == "VB":
                self.inf += 1
                self.active += 1
                time[4] = False  # set infinitive to False to check the next tense
            else:
                time[4] = False  # set infinitive to False to check the next tense
            time, prev_token, prev_pos = self.__get_time(time, token, pos)
        return time, prev_token, prev_pos

    def __get_participle(self, time, prev_token, prev_pos, token, pos):
        """ The participle can have different forms. The function checks which tense could be possible and if one tense
        is recognized, it will be incremented. At the end of the function the current token and current pos tag (xpos)
         are set to the previous ones. Possible tenses are:
        Present Participle -> Active: VBG; Passive: being + VBN
        Perfect Participle -> Active: having + VBN; Passive: having + been + VBN
        :param time: time-array that contains boolean. The current tense has the value True (here participle)
        :param prev_token: the previous token to identify the possible tenses
        :param prev_pos: the previous pos tag (xpos) to identify the possible tenses
        :param token: the current token
        :param pos: the current pos tag (xpos)
        :return: time-array, the previous token and the previous pos tag (xpos)
        """
        if prev_token == "being" and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        elif prev_token == "having" and pos == "VBN":
            self.was_perf = True
            prev_token, prev_pos = token, pos
        elif prev_pos == "VBN" and self.was_perf and pos == "VBN":
            self.is_passive = True
            prev_token, prev_pos = token, pos
        else:
            if prev_pos == "VBG":
                self.gerund += 1
                self.active += 1
            elif prev_pos == "VBN":
                if self.is_passive:
                    if self.was_perf:
                        self.perf_part += 1
                    else:
                        self.gerund += 1
                    self.is_passive = False
                    self.passive += 1
                else:
                    self.perf_part += 1
                    self.active += 1
            self.was_perf = False
            time[5] = False  # set participle to False to check the next tense
            time, prev_token, prev_pos = self.__get_time(time, token, pos)
        return time, prev_token, prev_pos

    def __get_last(self, time, prev_pos):
        """ Checks for the last token the tense
        :param time: time-array that contains boolean. The current tense has the value True
        :param prev_pos: the previous pos tag (xpos) to identify the possible tenses
        """
        if time[0]:  # Present
            if prev_pos in ["VBZ", "VB", "VBP"]:
                self.sim_pres += 1
                self.active += 1
            elif prev_pos == "VBN":
                if self.is_passive:
                    if self.was_prog:
                        self.pres_prog += 1
                    elif self.was_perf:
                        self.pres_perf += 1
                    else:
                        self.sim_pres += 1
                    self.passive += 1
                else:
                    self.pres_perf += 1
                    self.active += 1
            elif prev_pos == "VBG":
                if self.was_perf:
                    self.pres_perf_prog *= 1
                    self.active += 1
                else:
                    self.pres_prog += 1
                    self.active += 1
        elif time[1]:  # Past
            if prev_pos in "VBD, VB":
                self.sim_pas += 1
                self.active += 1
            elif prev_pos == "VBN":
                if self.is_passive:
                    if self.was_prog:
                        self.pas_prog += 1
                    elif self.was_perf:
                        self.pas_perf += 1
                    else:
                        self.sim_pas += 1
                    self.passive += 1
                else:
                    self.pas_perf += 1
                    self.active += 1
            elif prev_pos == "VBG":
                if self.was_perf:
                    self.pas_perf_prog += 1
                    self.active += 1
                else:
                    self.pas_prog += 1
                    self.active += 1
        elif time[2]:  # Future
            if prev_pos in ["MD", "VB"]:
                self.will += 1
                self.active += 1
            elif prev_pos == "VBN":
                if self.is_passive:
                    self.will += 1
                    self.passive += 1
                else:
                    self.fu_perf += 1
                    self.active += 1
            elif prev_pos == "VBG":
                if self.was_prog:
                    self.fu_prog += 1
                else:
                    self.fu_perf_prog += 1
                self.active += 1
        elif time[3]:  # Condition
            if prev_pos in ["MD", "VB"]:  # signalise would
                self.cond_sim += 1
                self.active += 1
            elif prev_pos == "VBN":
                if self.is_passive:
                    if self.was_perf:
                        self.cond_perf += 1
                    else:
                        self.cond_sim += 1
                    self.passive += 1
                else:
                    self.cond_perf += 1
                    self.active += 1
            elif prev_pos == "VBG":
                if self.was_perf:
                    self.cond_perf_prog += 1
                    self.active += 1
                else:
                    self.cond_prog += 1
                    self.active += 1
        elif time[4]:  # Infinitive
            if prev_pos == "VB":
                self.inf += 1
                self.active += 1
            elif prev_pos == "VBN":
                if self.is_passive:
                    if self.was_perf:
                        self.perf_inf += 1
                    else:
                        self.inf += 1
                    self.passive += 1
                else:
                    self.perf_inf += 1
                    self.active += 1
        elif time[5]:  # Participle
            if prev_pos == "VBG":
                self.gerund += 1
                self.active += 1
            elif prev_pos == "VBN":
                if self.is_passive:
                    if self.was_perf:
                        self.perf_part += 1
                    else:
                        self.gerund += 1
                    self.passive += 1
                else:
                    self.perf_part += 1
                    self.active += 1

    def __reset_help(self):
        """ Resets the help variables to the default value to make the correct decision for the next sentence
        """
        self.is_passive = False
        self.was_prog = False
        self.was_perf = False

    def calculate_tenses(self, part):
        """ Calculates all tenses for a sentence part/ line part
        :param part: the part of a sentence/line
        """
        prev_token = ''
        prev_pos = ''
        time = [False, False, False, False, False, False]  # present, past, future, condition, infinitive, Participle
        for token, pos in part:  # iterates through all tokens
            # if there is no current time, the method checks which time is possible
            if not (time[0] or time[1] or time[2] or time[3] or time[4] or time[5]):
                time, prev_token, prev_pos = self.__get_time(time, token, pos)
            elif time[0]:  # current Present
                time, prev_token, prev_pos = self.__get_present(time, prev_token, prev_pos, token, pos)
            elif time[1]:  # current Past
                time, prev_token, prev_pos = self.__get_past(time, prev_token, prev_pos, token, pos)
            elif time[2]:  # current Future
                time, prev_token, prev_pos = self.__get_future(time, prev_token, prev_pos, token, pos)
            elif time[3]:  # current  Conditional
                time, prev_token, prev_pos = self.__get_conditional(time, prev_token, prev_pos, token, pos)
            elif time[4]:  # current Infinitive
                time, prev_token, prev_pos = self.__get_infinitive(time, prev_token, prev_pos, token, pos)
            elif time[5]:  # current Participle
                time, prev_token, prev_pos = self.__get_participle(time, prev_token, prev_pos, token, pos)

        self.__get_last(time, prev_pos)
        self.__reset_help()

    def process_tenses(self, sentence):
        """ Processes the sentence/line to calculate the tenses and stores them
        :param sentence: the current sentence/line
        """
        for part in sentence:
            self.calculate_tenses(part)
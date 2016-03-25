#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <windows.h>
#include <stdexcept>

using namespace std;

#define maxPlayers 8 //the maximum number of players which can be in one game

#define maxLayerSize 12 //the largest number of nodes in one layer
#define inputLayerSize 12 //the number of input variables in the neural network, including a bias input. To increase layer sizes use the addNewWeights function
#define hiddenLayerSize 9 //the number of nodes in the hidden layer of the neural network, including a bias input
#define outputLayerSize 3 //the number of output variables in the neural network, this is always 2 for deciding fold/check/call/raise and amount raised
#define numberLayers 4

#define familyCount 5 //the number of families in the evolutionary algorithm
#define familyMembers 10 //the number of members in each family in the evolutionary algorithm

//below are means and ranges for input variables which are used to normalize inputs
#define handStrengthMean 0.475
#define handStrengthRange 0.45
#define logPotMean 3.190
#define logPotRange 5.583
#define logCallValueMean 2.284
#define logCallValueRange 4.549
#define logExistingBetMean 2.479
#define logExistingBetRange 4.842
#define roundNumberMean 2.5
#define roundNumberRange 3.0
#define playersActiveMean 4.299
#define playersActiveStDev 1.640
#define initialNumberPlayersMean 5.0
#define initialNumberPlayersRange 6.0
#define logOpponentsInitialChipsMean 4.205
#define logOpponentsInitialChipsRange 1.75
#define logMyInitialChipsMean 3.851
#define logMyInitialChipsRange 2.905
#define logFirstTopRaiseMean 6.54
#define logFirstTopRaiseRange 5.92
#define logSecondTopRaiseMean 6.005
#define logSecondTopRaiseRange 5.15
#define logThirdTopRaiseMean 5.415
#define logThirdTopRaiseRange 4.55

//calculate winProb so that its winProbConfidence confidence interval is within winProbAccuracy percentiles of its sample mean
#define winProbAccuracy 0.05
#define winProbConfidence 0.95

//calculate the family fitness rankings to geneFitnessRankingConfidence level of confidence
#define geneFitnessRankingConfidence 0.95
#define geneFitnessRankingAccuracy 0.25 //if geneFitnessRankingAccuracy = 0.125 the top 12.5% of members which get killed and the bottom 12.5% of members which stay alive may be mixed up

unsigned globalSeed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator (globalSeed);

double RationalApproximation(double t)
{   //RationalApproximation returns a value t used in calculating the inverse of the CDF of a normal distribution
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) /
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

double NormalCDFInverse(double p)
{   //NormalCDFInverse takes a probability and returns the z score corresponding to that probability
    // Abramowitz and Stegun formula 26.2.23.
    if (p <= 0.0 || p >= 1.0)
    {
        std::stringstream os;
        os << "Invalid input argument (" << p
           << "); must be larger than 0 but less than 1.";
        throw std::invalid_argument( os.str() );
    }

    // See article above for explanation of this section.
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}

int sortCards(float cards[7], float suits[7])
{   //sorts cards in descending order and sorts suits accordingly
    int i;
    for(i = 0; i < 7; i ++)
    {
        for(int j = 0; j < 7; j ++)
        {
            if(cards[i] > cards[j])
            {
                float temp = cards[i];
                cards[i] = cards[j];
                cards[j] = temp;
                float temp2 = suits[i];
                suits[i] = suits[j];
                suits[j] = temp2;
            }
        }
    }
    return 0; //the array cards[] is sorted but not returned
}

double checkHighCard(float cards[7])
{   //calculate the hand score for a high card
    double handScore = 1;
    for(int i = 0; i < 5; i ++)
    {
        handScore = handScore + cards[i] * pow(0.01, i + 1);
    }
    return handScore;
}

double checkPair(float cards[7])
{   //check if hand has a pair and return hand score
    double handScore = 0;
    for(int i = 0; i < 6; i ++)
    {
        if(cards[i] == cards[i + 1])
        {
            handScore = 2 + cards[i] / 100;
            i = 6;
        }
    }
    if(handScore != 0){
        for(int j = 0; j < 5; j ++){
            handScore = handScore + (cards[j] / 100) * pow(0.01, j + 1);
        }
    }
    return handScore;
}

double checkTwoPair(float cards[7])
{   //check if hand has two pairs and return hand score
    //cards must be sorted in descending order
    double handScore = 0;
    for(int i = 0; i < 4; i ++)
    {
        if(cards[i] == cards[i + 1])
        {
            for(int j = i + 2; j < 6; j ++)
            {
                if(cards[j] == cards[j + 1])
                {
                    //cards[i] is the highest pair value, cards[j] is the other pair value
                    handScore = 3 + cards[i] / 100 + cards[j] / 10000; //first two decimal places indicate highest pair, next two decimal places indicate other pair
                    for(int k = 0; k < 5; k ++)
                    {
                        if((cards[k] != cards[i]) && (cards[k] != cards[j]))
                        {
                            handScore += cards[k] / 1000000; //fifth and sixth decimal places indicate high card
                            //once twopair has been found set i, j and k to their max so that search for twopair ends
                            i = 4;
                            j = 6;
                            k = 5;
                        }
                    }
                }
            }
        }
    }
    return handScore;
}

double checkThreeOfAKind(float cards[7])
{   //check if hand has three of a kind and return hand score
    //cards must be sorted in descending order
    double handScore = 0;
    for(int i = 0; i < 5; i ++)
    {
        if((cards[i] == cards[i + 1]) && (cards[i] == cards[i + 2]))
        {
            handScore = 4 + cards[i] / 100;
            i = 5; //end search for triplet of cards
        }
    }
    if(handScore != 0)
    {
        for(int j = 0; j < 5; j ++)
        {
            handScore += (cards[j] / 100)*pow(0.01, j + 1); //add top 5 cards to the decimal places of handScore to indicate high cards
        }
    }
    return handScore;
}

double checkStraight(float cards[7])
{   //check if hand has a straight and return hand score
    float handScore = 0;
    float topCard = 0;
    for(int i = 0; i < 3; i ++)
    {
        //when i is 3 or greater there cannot be 4 cards in order which follow cards[i]
        topCard = cards[i];
        for(int j1 = i + 1; j1 < 4; j1 ++)
        {
            if(topCard == (cards[j1] + 1))
            {
                for(int j2 = j1 + 1; j2 < 5; j2 ++)
                {
                    if(topCard == (cards[j2] + 2))
                    {
                        for(int j3 = j2 + 1; j3 < 6; j3 ++)
                        {
                            if(topCard == (cards[j3] + 3))
                            {
                                for(int j4 = j3 + 1; j4 < 7; j4 ++)
                                {
                                    if(topCard == (cards[j4] + 4))
                                    {
                                        handScore = 5 + topCard / 100;
                                        i = 3;
                                        j1 = 4;
                                        j2 = 5;
                                        j3 = 6;
                                        j4 = 7;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return handScore;
}

double checkFlush(float cards[7], float suits[7])
{   //check if hand has a flush and return hand score
    //suits must be sorted in descending order with cards sorted accordingly
    double handScore = 0;
    double temp = 0;
    float clubs = 0, diamonds = 0, hearts = 0, spades = 0; //number of each suit
    float flushSuit = 0;
    int j;
    for(int i = 0; i < 7; i ++)
    {
        if(suits[i] == 1)
        {
            clubs += 1;
        }
        else if(suits[i] == 2)
        {
            diamonds += 1;
        }
        else if(suits[i] == 3)
        {
            hearts += 1;
        }
        else
        {
            spades += 1;
        }
    }
    //check what suit  has flush
    if(clubs > 4)
    {
        handScore = 6;
        flushSuit = 1;
    }
    else if(diamonds > 4)
    {
        handScore= 6;
        flushSuit = 2;
    }
    else if(hearts > 4)
    {
        handScore = 6;
        flushSuit = 3;
    }
    else if(spades > 4)
    {
        handScore = 6;
        flushSuit = 4;
    }
    if(handScore == 6)
    {
        //add first high card
        for(j = 0; j < 3; j ++)
        {
            if(suits[j] == flushSuit)
            {
                temp = cards[j];
                handScore += temp / 100;
                break;
            }
        }
        //add second high card
        for(j = j + 1; j < 4; j ++)
        {
            if(suits[j] == flushSuit)
            {
                temp = cards[j];
                handScore += temp / 10000;
                break;
            }
        }
        //add third high card
        for(j = j + 1; j < 5; j ++)
        {
            if(suits[j] == flushSuit)
            {
                temp = cards[j];
                handScore += temp / 1000000;
                break;
            }
        }
        //add fourth high card
        for(j = j + 1; j < 6; j ++)
        {
            if(suits[j] == flushSuit)
            {
                temp = cards[j];
                handScore += temp / 100000000;
                break;
            }
        }
        //add fifth high card
        for(j = j + 1; j < 7; j ++)
        {
          if(suits[j] == flushSuit)
          {
            temp = cards[j];
            handScore += temp / 10000000000;
            break;
          }
        }
    }
    return handScore;
}

double checkFullHouse(float cards[7])
{   //check if hand has a full house and return hand score
    double handScore = 0;
	for(int i = 0; i < 5; i ++)
    {
		//check for triples
		if((cards[i] == cards[i + 1]) && (cards[i] == cards[i + 2]))
		{
			//check for doubles higher than triples
			for(int j = 0; j < (i - 1); j ++)
			{
				if((cards[j] == cards[j + 1]))
				{
					handScore = 7 + (cards[i] / 100) + (cards[j] / 10000);
                }
            }
			//check for doubles lower than triples
			for(int j = i + 3; j < 7; j ++)
			{
				if((cards[j] == cards[j + 1]))
				{
					handScore = 7 + (cards[i] / 100) + (cards[j] / 10000);
					i = 5;
                    j = 7;
                }
            }
        }
    }
    return handScore;
}

double checkFourOfAKind(float cards[7])
{   //check for four of a kind and return hand score
    double handScore = 0;
    for(int i = 0; i < 5; i ++)
    {
        if((cards[i] == cards[i + 1]) && (cards[i] == cards[i + 2]) && (cards[i] == cards[i + 3]))
        {
            handScore = 8 + cards[i] / 100;
            i = 5;
        }
    }
    if(handScore != 0)
    {
        for(int j = 0; j < 5; j ++)
        {
            handScore += (cards[j] / 100) * pow(0.01, j + 1);
        }
    }
    return handScore;
}

double checkStraightFlush(float cards[7], float suits[7])
{   //check if hand has a straight flush and return hand score
    float handScore = 0;
    float topCard = 0;
    float topSuit = 0; //the suit of the top card
    for(int i = 0; i < 3; i ++)
    {
        topCard = cards[i];
        topSuit = suits[i];
        for(int j1 = i + 1; j1 < 4; j1 ++)
        {
            if((topCard == (cards[j1] + 1)) && (topSuit == suits[j1]))
            {
                for(int j2 = j1 + 1; j2 < 5; j2 ++)
                {
                    if((topCard == (cards[j2] + 2)) && (topSuit == suits[j2]))
                    {
                        for(int j3 = j2 + 1; j3 < 6; j3 ++)
                        {
                            if((topCard == (cards[j3] + 3)) && (topSuit == suits[j3]))
                            {
                                for(int j4 = j3 + 1; j4 < 7; j4 ++)
                                {
                                    if((topCard == (cards[j4] + 4)) && (topSuit == suits[j4]))
                                    {
                                        handScore = 9 + topCard / 100;
                                        i = 3;
                                        j1 = 4;
                                        j2 = 5;
                                        j3 = 6;
                                        j4 = 7;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return handScore;
}

double getHandScore(float cards[7], float suits[7])
{   //returns the value of the best hand from 7 cards
    sortCards(cards, suits); //put the cards in order of highest to lowest
    double handScore = 0;
    handScore = checkStraightFlush(cards, suits);
    if(handScore == 0) //if handScore didn't change then check for the next best hand
        {
        handScore = checkFourOfAKind(cards);
        if(handScore == 0)
        {
            handScore = checkFullHouse(cards);
            if(handScore == 0)
            {
                handScore = checkFlush(cards, suits);
                if(handScore == 0)
                {
                    handScore = checkStraight(cards);
                    if(handScore == 0)
                    {
                        handScore = checkThreeOfAKind(cards);
                        if(handScore == 0)
                        {
                            handScore = checkTwoPair(cards);
                            if(handScore == 0)
                            {
                                handScore = checkPair(cards);
                                if(handScore == 0)
                                {
                                    handScore = checkHighCard(cards);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return handScore;
}

float dealCard(float existingCards[5 + (maxPlayers * 2)], float existingSuits[5 + (maxPlayers * 2)], float newCard[2])
{   //dealCard modifies newCard[2] to enter a new card and suit
    int cardIndex, cardNumber, suitNumber;
    int uniqueness = 0; //0 is not unique card, 1 is unique

    //loop until a unique card is found
    while(!uniqueness)
    {
        std::uniform_int_distribution<int> cardPack(0, 51);
        cardIndex = cardPack (generator);
        cardNumber = (cardIndex % 13) + 2;
        suitNumber = (cardIndex % 4) + 1;
        uniqueness = 1;
        for(int j = 0; j < 52; j ++)
        {
            if(existingCards[j] == 0)
            {
                j = 52; //if all cards have been checked then end loop
            }
            else if((cardNumber == existingCards[j]) && (suitNumber == existingSuits[j]))
            {
                uniqueness = 0;
                j = 52; //if card is not unique then end search
            }
        }
        if(uniqueness == 1)
        {
            newCard[0] = cardNumber;
            newCard[1] = suitNumber;
        }
    }
    return 0;
}

int countPlayers(int playersKnockedOut[maxPlayers])
{   //countPlayers counts how many players are not knocked out
    int numberPlayers = 0;
    for(int i = 0; i < maxPlayers; i ++)
    {
        if(!playersKnockedOut[i])
        {
            numberPlayers ++;
        }
    }
    return numberPlayers;
}

double normZScore(double probability, double mean, double stDev)
{   //calculate the z score for a given cumulative probability of a normal distribution
    double standardZ = NormalCDFInverse(probability);
    double z = (standardZ * stDev) + mean;
    return z;
}

double normCDF(double z, double mean, double stDev)
{   //calculate the cumulative probability for a given z score of a normal distribution
    double standardZ = (z - mean) / stDev;
    double cumulativeProbability = 1 / (exp((-15.565 * standardZ) + 111 * atan(0.12585 * standardZ)) + 1);
    return cumulativeProbability;
}

//winProbRequiredSamples only used in the case where handStrengths in game have a normal distribution
/*
double winProbRequiredSamples(double prob)
{   //winProbRequiredSamples calculates the number of samples required to calculate the probability of cards winning to an acceptable level of accuracy
    int requiredSamples;
    double normalizedProb = ((prob - handStrengthMean) / handStrengthRange);
    //calculate the required sample size to estimate winProb to within winProbAccuracy percentiles of the distribution of winProbs. This is calculated with winProbConfidence level of confidence
    double confidenceZScore = normZScore((1 + winProbConfidence) / 2, 0, 1);
    double sampleMeanCDF = normCDF(normalizedProb, 0, 1);
    if(prob < handStrengthMean)
    {   //the maximum of the confidence interval for winProb is less than winProbAccuracy percentiles from the sample mean
        double CImaximum = normZScore(sampleMeanCDF + winProbAccuracy, handStrengthMean, handStrengthStDev); // the minimum of the confidence interval for the sample mean
        requiredSamples = pow(confidenceZScore * sqrt((prob) * (1 - prob)) / (CImaximum - prob), 2.0);
    }
    else
    {   //the minimum of the confidence interval for winProb is less than winProbAccuracy percentiles from the sample mean
        double CIminimum = normZScore(sampleMeanCDF - winProbAccuracy, handStrengthMean, handStrengthStDev); // the maximum of the confidence interval for the sample mean
        requiredSamples = pow(confidenceZScore * sqrt((prob) * (1 - prob)) / (prob - CIminimum), 2.0);
    }
    return requiredSamples;
}
*/

double winProb(float holeCards[2], float holeSuits[2], float communityCards[5], float communitySuits[5], double playersActive)
{   //winProb calculates the probability that the set of cards will beat all remaining players assuming those players have random hands
    //community cards are 0,0 if not yet dealt

    double probability, prob; //prob is the chance of beating one player, probability is the chance of beating all players
    float wins = 0;
    double myHandValue, oppHandValue; //this hand's and the opponent's hand's value
    float samples = 0; //number of times future cards have been simulated
    int maxSamples = 400; //the maximum number of samples which should be done before returning hand strength
    int commCards; //the number of Community Cards;
    float deal[2];
    float oppCards[7] = {0}, oppSuits[7] = {0}; //opponent's cards and suits
    float existingCards[9] = {0}, existingSuits[9] = {0}; //limited to 5 community cards, my cards and opponent's cards. Other players' hole cards are unseen and so are not considered as being dealt already
    float myCards[7] = {0}, mySuits[7] = {0}; //the cards of the player whose winprob is being calculated (including community cards)
    float commonCards[5] = {0}, commonSuits[5] = {0};

    //calculate minimum number of samples based off Wilson score method under the worst case scenario where handStrength = 0.5
    double confidenceZScore = normZScore(0.5 + (winProbConfidence / 2), 0, 1);

    //commented out code is for the case when handStrengths have a normal distribution
    ///double confidenceMaximum = normZScore(winProbAccuracy, handStrengthMean, handStrengthStDev);
    ///int minSamples = 1 + (pow(confidenceZScore, 2) / confidenceMaximum) - confidenceZScore;

    int minSamples = 1 + pow(confidenceZScore, 2.0) * (1 / (winProbAccuracy * handStrengthRange) - 1);

    if(minSamples > maxSamples)
    {
        minSamples = maxSamples;
    }

    //loop for number of outcomes explored
    while(samples <= minSamples)
    {
        samples ++;
        //each time redefine card arrays which get altered during calculation of hand strength

        //put community cards into new array
        for(int i = 0; i < 5; i ++)
        {
            commonCards[i] = communityCards[i];
            commonSuits[i] = communitySuits[i];
        }

        //determine how many community cards there are
        for(commCards = 0; commCards < 5; commCards ++)
        {
            if(commonCards[commCards] == 0)
            {
                break;
            }
        }

        //put hole cards and community cards into my cards and existing cards
        for(int i = 0; i < 2; i ++)
        {
            existingCards[i] = holeCards[i];
            existingSuits[i] = holeSuits[i];
            myCards[i] = holeCards[i];
            mySuits[i] = holeSuits[i];
        }

        for(int i = 0; i < commCards; i ++)
        {
            existingCards[i + 2] = commonCards[i];
            existingSuits[i + 2] = commonSuits[i];
            myCards[i + 2] = commonCards[i];
            mySuits[i + 2] = commonSuits[i];
        }

        //generate remaining community cards
        for(int i = commCards; i < 5; i ++)
        {
            dealCard(existingCards, existingSuits, deal);
            existingCards[i + 2] = deal[0];
            existingSuits[i + 2] = deal[1];
            commonCards[i] = deal[0];
            commonSuits[i] = deal[1];
        }

        //generate opponent's cards
        for(int i = 0; i < 2; i ++)
        {
            dealCard(existingCards, existingSuits, deal);
            existingCards[i + 7] = deal[0];
            existingSuits[i + 7] = deal[1];
            oppCards[i] = deal[0];
            oppSuits[i] = deal[1];
        }

        //fill in my and opponent's cards from community cards
        for(int i = 0; i < 5; i ++){
            myCards[i + 2] = commonCards[i];
            mySuits[i + 2] = commonSuits[i];
            oppCards[i + 2] = commonCards[i];
            oppSuits[i + 2] = commonSuits[i];
        }

        //calculate and compare handscores
        myHandValue = getHandScore(myCards, mySuits);
        oppHandValue = getHandScore(oppCards, oppSuits);

        if(myHandValue > oppHandValue)
        {
            wins ++;
        }
        if(myHandValue == oppHandValue)
        {
            wins += 0.5;
        }

        prob = (wins / samples); //probability of beating one player

        if((samples == minSamples) && (prob != 0) && (prob != 1))
        {
            double confidenceZScoreSquared = pow(confidenceZScore, 2.0);
            double temp = samples;
            double samplesSquared = pow(temp, 2.0);
            double uncertainty; //the range of uncertainty of the true value of prob
            if(prob > 0.5)
            {
                uncertainty = (1 / (1 + (confidenceZScoreSquared / temp))) * (prob + (0.5 * confidenceZScoreSquared / temp) + confidenceZScore * pow(prob * (1 - prob) / temp + 0.25 * confidenceZScoreSquared / samplesSquared, 0.5)) - prob;
            }
            else
            {
                uncertainty = prob - (1 / (1 + (confidenceZScoreSquared / temp))) * (prob + (0.5 * confidenceZScoreSquared / temp) - confidenceZScore * pow(prob * (1 - prob) / temp + 0.25 * confidenceZScoreSquared / samplesSquared, 0.5));
            }

            //commented out code is for the case when handstrengths have a normal distribution
            ///minSamples = winProbRequiredSamples(prob);

            if(uncertainty > (winProbAccuracy * handStrengthRange))
            {   //if the uncertainty is above the required level increase the number of samples required by 20
                minSamples += 20;
            }
        }
    }

    probability = pow(prob, playersActive - 1); //probability of beating all players

    return probability;
}

int getHandStrengths(double handStrengths[maxPlayers], int roundNumber, int folds[maxPlayers], float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float communityCards[5], float communitySuits[5])
{
    std::uniform_real_distribution<double> uniformdistribution(0.0,1.0);
    float holeCards[2];
    float holeSuits[2];
    for(int i = 0; i < maxPlayers; i ++)
    {
        if(!folds[i])
        {
            holeCards[0] = playerCards[i][0];
            holeCards[1] = playerCards[i][1];
            holeSuits[0] = playerSuits[i][0];
            holeSuits[1] = playerSuits[i][1];

            handStrengths[i] = winProb(holeCards, holeSuits, communityCards, communitySuits, 2);
        }
    }
    return 0;
}

int printInfo(int playersKnockedOut[maxPlayers], int startPosition, string playerNames[maxPlayers], int chips[maxPlayers], int aiPlayers[maxPlayers])
{   //print players names, dealer's name, players chips;
    int position;
    int numberPlayers = countPlayers(playersKnockedOut);
    cout << "There are " << numberPlayers << " players" << endl;
    cout << "The players in the game are:" << endl;
    for(int i = startPosition; i < maxPlayers + startPosition; i ++)
    {
        position = i % maxPlayers;
        if(playersKnockedOut[position])
        {
            cout << playerNames[position] << "\t" << chips[position];
            if(i == startPosition)
            {
                cout << " (dealer)";
            }
            if(aiPlayers[position] == 1)
            {
                cout << " (computer)";
            }
            cout << endl;
        }
    }
    return 0;
}

int getInfo(string playerNames[maxPlayers], int aiPlayers[maxPlayers], int chips[maxPlayers])
{   //getInfo modifies playerNames, AIplayers and chips arrays and returns the number of players
    int numberPlayersEntry;
    cout << "How many players are there?" << endl;
    cin >> numberPlayersEntry;
    cout << "Enter the dealer's name" << endl;
    cin >> playerNames[0];
    for(int i = 0; i < numberPlayersEntry; i ++)
    {
        cout << "How many chips does " << playerNames[i] << " have?" << endl;
     	cin >> chips[i];
        cout << "Is " << playerNames[i] << " a computer? (enter 1 for yes, 0 for no)" << endl;
     	cin >> aiPlayers[i];
     	cout << endl;
        if(i + 1 < numberPlayersEntry)
        {
            cout << "Enter the next player's name" << endl;
            cin >> playerNames[i + 1];
        }
    }
    return numberPlayersEntry;
}

int checkDetails(int playersKnockedOut[maxPlayers], int startPosition, string playerNames[maxPlayers], int aiPlayers[maxPlayers], int chips[maxPlayers])
{   //checks if the players' names, order, and chip stacks are correct
    int infoCheck = 0; //when infoCheck is 1 the information stored is correct
    int numberPlayers;
    while(!infoCheck)
    {
        //Before round starts print the information to check if it is correct
        printInfo(playersKnockedOut, startPosition, playerNames, chips, aiPlayers);
        //ask if info is correct, if not request names and chips
        cout << "Is information correct? (enter 1 if correct)" << endl;
        cin >> infoCheck;
        if(infoCheck != 1)
        {
            numberPlayers = getInfo(playerNames, aiPlayers, chips);
        }
    }
    return 0;
}

char getSuitLetter(float suitNumber)
{   //getSuitLetter converts a suit's number to a card's suit to display
    char suit = ' ';
    if(suitNumber == 1){
        suit = '\5';
    }
    else if(suitNumber == 2){
        suit = '\4';
    }
    else if(suitNumber == 3){
        suit = '\3';
    }
    else if(suitNumber == 4){
        suit = '\6';
    }
    return suit;
}

char getCardLetter(float cardNumber)
{   //getCardLetter converts trick card numbers into letters
    char card = '0';
    if(cardNumber == 14){
        card = 'A';
    }
    else if(cardNumber == 13){
        card = 'K';
    }
    else if(cardNumber == 12){
        card = 'Q';
    }
    else if(cardNumber == 11){
        card = 'J';
    }
    return card;
}

float getCardNumber()
{   //getCardNumber requests the user to enter a card's number until receiving a valid string and converts this string to a integer card number
    string card;
    int invalidCard = 1;
    float cardNumber;
    while(invalidCard)
    {   //while the card entered is invalid repeat loop
        cin >> card;
        if((card == "A") || (card == "a") || (card == "Ace") || (card == "ace") || (card == "ACE") || (card == "14"))
        {
            cardNumber = 14;
            invalidCard = 0;
        }
        else if((card == "K") || (card == "k") || (card == "King") || (card == "king") || (card == "KING") || (card == "13"))
        {
            cardNumber = 13;
            invalidCard = 0;
        }
        else if((card == "Q") || (card == "q") || (card == "Queen") || (card == "queen") || (card == "QUEEN") || (card == "12"))
        {
            cardNumber = 12;
            invalidCard = 0;
        }
        else if((card == "J") || (card == "j") || (card == "Jack") || ( card == "jack") || (card == "JACK") || (card == "11"))
        {
            cardNumber = 11;
            invalidCard = 0;
        }
        else if(card == "10")
        {
            cardNumber = 10;
            invalidCard = 0;
        }
        else if(card == "9")
        {
            cardNumber = 9;
            invalidCard = 0;
        }
        else if(card == "8")
        {
            cardNumber = 8;
            invalidCard = 0;
        }
        else if(card == "7")
        {
            cardNumber = 7;
            invalidCard = 0;
        }
        else if(card == "6")
        {
            cardNumber = 6;
            invalidCard = 0;
        }
        else if(card == "5")
        {
            cardNumber = 5;
            invalidCard = 0;
        }
        else if(card == "4")
        {
            cardNumber = 4;
            invalidCard = 0;
        }
        else if(card == "3")
        {
            cardNumber = 3;
            invalidCard = 0;
        }
        else if(card == "2")
        {
            cardNumber = 2;
            invalidCard = 0;
        }
        else
        {
            cout << "Enter a valid card number" << endl;
        }
    }
    return cardNumber;
}

float getSuitNumber()
{   //getSuitNumber requests the user to enter a card's suit until receiving a valid string and converts this string to a integer suit number
    string suit;
    int invalidSuit = 1;
    float suitNumber;
    while(invalidSuit)
    {
        cin >> suit;
        if((suit == "1")||(suit == "c")||(suit == "C")||(suit == "club")||(suit == "clubs"))
        {
            suitNumber = 1;
            invalidSuit = 0;
        }
        else if((suit == "2")||(suit == "d")||(suit == "D")||(suit == "diamond")||(suit == "diamonds"))
        {
            suitNumber = 2;
            invalidSuit = 0;
        }
        else if((suit == "3")||(suit == "h")||(suit == "H")||(suit == "heart")||(suit == "hearts"))
        {
            suitNumber = 3;
            invalidSuit = 0;
        }
        else if((suit == "4")||(suit == "s")||(suit == "S")||(suit == "spade")||(suit == "spades"))
        {
            suitNumber = 4;
            invalidSuit = 0;
        }
        else
        {
            cout << "Enter a valid suit" << endl;
        }
    }
    return suitNumber;
}

int showCards(int position, float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2])
{   //showCards prints the hole cards of a single player
    for(int i = 0; i < 2; i++)
    {
        if(playerCards[position][i] != 0)
        {
            char temp = getSuitLetter(playerSuits[position][i]);
            if(playerCards[position][i] > 10)
            {
                char temp2 = getCardLetter(playerCards[position][i]);
                cout << temp2 << " " << temp << endl;
            }
            else
            {
                int temp2 = playerCards[position][i];
                cout << temp2 << " " << temp << endl;
            }
        }
    }
    //end many lines so that next player can't see previous player's cards
    cout << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl;
    return 0;
}

int autoDeal(int roundNumber, int numberPlayers, int folds[maxPlayers], float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float existingCards[5 + (maxPlayers * 2)], float existingSuits[5 + (maxPlayers * 2)], float communityCards[5], float communitySuits[5])
{   //autoDeal hole cards or community cards for this round
    float newCard[2] = {0,0};
    if(roundNumber == 1)
    {   //Deal two cards to each person and update existingcards[21]
        for(int i = 0; i < maxPlayers; i ++)
        {
            if(!folds[i])
            {
                for(int j = 0; j < 2; j ++)
                {
                    dealCard(existingCards, existingSuits, newCard);
                    playerCards[i][j] = newCard[0];
                    playerSuits[i][j] = newCard[1];
                    existingCards[(2 * i) + j] = newCard[0];
                    existingSuits[(2 * i) + j] = newCard[1];
                }
            }
        }
    }
    if(roundNumber == 2)
    {
        for(int i = 0; i < 3; i ++)
        {
            dealCard(existingCards, existingSuits, newCard);
            communityCards[i] = newCard[0];
            communitySuits[i] = newCard[1];
            existingCards[(numberPlayers * 2) + i] = newCard[0];
            existingSuits[(numberPlayers * 2) + i] = newCard[1];
        }
        communityCards[3] = 0;
        communityCards[4] = 0;
        communitySuits[3] = 0;
        communitySuits[4] = 0;
    }
    if(roundNumber == 3)
    {
        dealCard(existingCards, existingSuits, newCard);
        communityCards[3] = newCard[0];
        communitySuits[3] = newCard[1];
        existingCards[(numberPlayers * 2) + 3] = newCard[0];
        existingSuits[(numberPlayers * 2) + 3] = newCard[1];
        communityCards[4] = 0;
        communitySuits[4] = 0;
    }
    if(roundNumber == 4)
    {
        dealCard(existingCards, existingSuits, newCard);
        communityCards[4] = newCard[0];
        communitySuits[4] = newCard[1];
        existingCards[(numberPlayers * 2) + 4] = newCard[0];
        existingSuits[(numberPlayers * 2) + 4] = newCard[1];
    }
    return 0;
}

int deal(int roundNumber, int dealerPosition, int manualDealing, int numberPlayers, int trainingMode, int aiPlayers[maxPlayers], int playersKnockedOut[maxPlayers], string playerNames[maxPlayers], int folds[maxPlayers], float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float existingCards[5 + (maxPlayers * 2)], float existingSuits[5 + (maxPlayers * 2)], float communityCards[5], float communitySuits[5])
{   //deal generates random cards for play if manualDealing is off or requests the new cards dealt from a real pack if manualDealing is on
    if(manualDealing == 0)
    {
        autoDeal(roundNumber, numberPlayers, folds, playerCards, playerSuits, existingCards, existingSuits, communityCards, communitySuits);
        //tell people their cards
        if(roundNumber == 1)
        {
            for(int k = 0; k < maxPlayers; k ++)
            {
                if((aiPlayers[k] == 0) && (playersKnockedOut[k] == 0))
                {
                    if(!trainingMode)
                    {
                        cout << playerNames[k] << " your cards are:" << endl << endl;
                        showCards(k, playerCards, playerSuits);
                        cout << "Enter anything to continue" << endl;
                        string temp;
                        cin >> temp;
                        cout << endl;
                    }
                }
            }
        }
    }
    if(manualDealing == 1)
    {
        string temp;
        if(roundNumber == 1)
        {
            for(int k = (dealerPosition + 1); k < (maxPlayers + dealerPosition + 1); k ++)
            {   //begin dealing from the dealer's left
                int dealPosition = k % maxPlayers;
                if(!playersKnockedOut[dealPosition])
                {
                    cout << "Enter " << playerNames[dealPosition] << "'s first card" << endl;
                    playerCards[dealPosition][0] = getCardNumber();
                    cout << "Enter that card's suit" << endl;
                    playerSuits[dealPosition][0] = getSuitNumber();
                    cout << "Enter " << playerNames[dealPosition] << "'s second card" << endl;
                    playerCards[dealPosition][1] = getCardNumber();
                    cout << "Enter that card's suit" << endl;
                    playerSuits[dealPosition][1] = getSuitNumber();
                }
            }
        }
        if(roundNumber == 2)
        {
            cout << "Enter the first flop card" << endl;
            communityCards[0] = getCardNumber();
            cout << "Enter that card's suit" << endl;
            communitySuits[0] = getSuitNumber();
            cout << "Enter the second flop card" << endl;
            communityCards[1] = getCardNumber();
            cout << "Enter that card's suit" << endl;
            communitySuits[1] = getSuitNumber();
            cout << "Enter the third flop card" << endl;
            communityCards[2] = getCardNumber();
            cout << "Enter that card's suit" << endl;
            communitySuits[2] = getSuitNumber();
            communityCards[3] = 0;
            communitySuits[3] = 0;
            communityCards[4] = 0;
            communitySuits[4] = 0;
        }
        if(roundNumber == 3)
        {
            cout << "Enter the turn card" << endl;
            communityCards[3] = getCardNumber();
            cout << "Enter that card's suit" << endl;
            communitySuits[3] = getSuitNumber();
            communityCards[4] = 0;
            communitySuits[4] = 0;
        }
        if(roundNumber == 4)
        {
            cout << "Enter the river card" << endl;
            communityCards[4] = getCardNumber();
            cout << "Enter that card's suit" << endl;
            communitySuits[4] = getSuitNumber();
        }
    }
    return 0;
}

int requestBet()
{   //requestBet asks the player for their bet
    int newbet;
    string betString;
    int validbet = 0;
    while(!validbet)
    {
        cin >> betString;
        validbet = 1; //assume bet is valid then test this
        istringstream(betString) >> newbet; //istringstream returns 0 if string has no number. This must not be confused with a bet of 0
        if(((newbet == 0) && (betString.at(0) != '0')) || (newbet < 0))
        {
            cout << "Enter a valid bet" << endl;
            validbet = 0;
        }
    }
    return newbet;
}

int getHumanBet(int position, int maxBet, int pot, string playerNames[maxPlayers], int chips[maxPlayers], int bets[maxPlayers], int calls[maxPlayers], int raises[maxPlayers])
{   //getHumanBet asks human player for bet and validates it
    int newBet;
    int betValue = 0; //betValue is the bet after modified for validity, betValue is -1 if invalid
    cout << endl << playerNames[position] << " has " << chips[position] << " chips" << " and has bet "  << bets[position] << " already" << endl;
    cout << "The bet to match is " << maxBet << " for a pot of " << pot << endl;
    int callValue = maxBet - bets[position];
    if(callValue < 0)
    {
        callValue = 0;
    }
    cout << playerNames[position] << ", the call value is " << callValue << ". How much are you betting?" << endl;
    //request bet
    newBet = requestBet();
    betValue = newBet;
    if(newBet != 0)
    {   //if player has not checked/folded
        //check if player cannot bet full amount
        if(chips[position] + bets[position] < maxBet)
        {
            //check that they have bet as much as they can
            if(newBet < chips[position])
            {
                cout << "You must bet all your money or fold" << endl;
                betValue = -1;
            }
        }
        else
        {
            //check if bet is valid
            if(((newBet + bets[position] >= maxBet) || (newBet == chips[position])) && (newBet <= chips[position]))
            {
                //if valid bet do nothing
            }
            else
            {
                betValue = -1; //bet is invalid
                if((chips[position] > newBet) && (newBet != 0))
                {
                    cout << "That's not enough" << endl;
                }
                if((chips[position] < newBet) && (newBet != 0))
                {
                    cout << "That's more than your chip stack" << endl;
                }
            }
        }
    }
    return betValue;
}

int sortWinners(double handScores[maxPlayers], int position[maxPlayers])
{   //sorts winners puts the positions of players in order of best to worst hands
    for(int i = 0; i < maxPlayers; i ++)
    {
        for(int j = 0; j < maxPlayers ; j ++)
        {
            if(handScores[i] > handScores[j])
            {
                double temp1 = handScores[i];
                handScores[i] = handScores[j];
                handScores[j] = temp1;
                int temp2 = position[i];
                position[i] = position[j];
                position[j] = temp2;
            }
        }
    }
    return 0;
}

int sortBetters(int bets[maxPlayers], int position[maxPlayers])
{   //sorts betters and puts the positions of betters in order of highest to lowest bets
    for(int i = 0; i < maxPlayers; i ++)
    {
        for(int j = 0; j < maxPlayers ; j ++)
        {
            if(bets[i] > bets[j])
            {
                double temp1 = bets[i];
                bets[i] = bets[j];
                bets[j] = temp1;
                int temp2 = position[i];
                position[i] = position[j];
                position[j] = temp2;
            }
        }
    }
    return 0;
}

int *sortIntArrayByDouble(int arrayToSort[], double sortArrayBy[], int arraySize, int sortAscending)
{   //sort arrayToSort in order of sortArrayBy. Sort in descending order if sortAscending is false
    if(sortAscending)
    {   //sort the array in ascending order
        for(int i = 0; i < arraySize; i ++)
        {
            for(int j = 0; j < arraySize ; j ++)
            {
                if(sortArrayBy[i] < sortArrayBy[j])
                {
                    double temp1 = sortArrayBy[i];
                    sortArrayBy[i] = sortArrayBy[j];
                    sortArrayBy[j] = temp1;
                    int temp2 = arrayToSort[i];
                    arrayToSort[i] = arrayToSort[j];
                    arrayToSort[j] = temp2;
                }
            }
        }
    }
    else{
        //sort the array in descending order
        for(int i = 0; i < arraySize; i ++)
        {
            for(int j = 0; j < arraySize ; j ++)
            {
                if(sortArrayBy[i] > sortArrayBy[j])
                {
                    double temp1 = sortArrayBy[i];
                    sortArrayBy[i] = sortArrayBy[j];
                    sortArrayBy[j] = temp1;
                    int temp2 = arrayToSort[i];
                    arrayToSort[i] = arrayToSort[j];
                    arrayToSort[j] = temp2;
                }
            }
        }
    }
    return arrayToSort;
}

int sortIntArrayByInt(int arrayToSort[], int sortArrayBy[], int arraySize, int sortAscending)
{   //sort arrayToSort in order of sortArrayBy. Sort in descending order if sortAscending is false
    if(sortAscending)
    {   //sort the array in ascending order
        for(int i = 0; i < arraySize; i ++)
        {
            for(int j = 0; j < arraySize ; j ++)
            {
                if(sortArrayBy[i] < sortArrayBy[j])
                {
                    double temp1 = sortArrayBy[i];
                    sortArrayBy[i] = sortArrayBy[j];
                    sortArrayBy[j] = temp1;
                    int temp2 = arrayToSort[i];
                    arrayToSort[i] = arrayToSort[j];
                    arrayToSort[j] = temp2;
                }
            }
        }
    }
    else{
        //sort the array in descending order
        for(int i = 0; i < arraySize; i ++)
        {
            for(int j = 0; j < arraySize ; j ++)
            {
                if(sortArrayBy[i] > sortArrayBy[j])
                {
                    double temp1 = sortArrayBy[i];
                    sortArrayBy[i] = sortArrayBy[j];
                    sortArrayBy[j] = temp1;
                    int temp2 = arrayToSort[i];
                    arrayToSort[i] = arrayToSort[j];
                    arrayToSort[j] = temp2;
                }
            }
        }
    }
    return 0;
}

int findWinners(int winnerPositions[maxPlayers], float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float communityCards[5], float communitySuits[5], int folds[maxPlayers])
{   //findwinners returns positions of best to worst hands
    //initially winnerPositions are in order of player not hand score
    double handScores[maxPlayers];
    for(int k = 0; k < maxPlayers; k ++)
    {
        winnerPositions[k] = k;
        handScores[k] = 0;
    }
    for(int position = 0; position < maxPlayers; position ++){
        //check if player is folded or knocked out, if not then find their handscore
        if(folds[position] == 0){
            //combine community cards and player’s cards
            float handScoreCards[7];
            float handScoreSuits[7];
            for(int j = 0; j < 5; j++){
                handScoreCards[j] = communityCards[j];
                handScoreSuits[j] = communitySuits[j];
            }
            handScoreCards[5] = playerCards[position][0];
            handScoreSuits[5] = playerSuits[position][0];
            handScoreCards[6] = playerCards[position][1];
            handScoreSuits[6] = playerSuits[position][1];
            //find player’s handscore
            handScores[position] = getHandScore(handScoreCards, handScoreSuits);
        }
	}
	sortWinners(handScores, winnerPositions);
    return 0;
}

float winnerChange(float selfCards[2], float selfSuits[2], float communityCards[5], float communitySuits[5], int roundNumber)
{  //winnerChange looks at if my hand beats other hands before and after new community cards, returns how often the winner changes
    int winBefore, winAfter; //0 if they win, 1 if I win
    float changes = 0;
    float samples = 5000; //made a float so that division can be done without truncation
    float averageChange;
    int commCards; //number of comm cards present
    int commCardsBefore;
    float existingCards[9] = {0,0,0,0,0,0,0,0,0};
    float existingSuits[9] = {0,0,0,0,0,0,0,0,0};
    float oppCards[7] = {0,0,0,0,0,0,0};
    float oppSuits[7] = {0,0,0,0,0,0,0};
    float myCards[7] = {0,0,0,0,0,0,0}; //the cards of the player who's winnerchange is being calculated
    float mySuits[7] = {0,0,0,0,0,0,0};
    float deal[2];
    double myHandValue, oppHandValue;
	//generate random hand
	//generate remaining communitycards for old and new
	//fill existingcards for old and new and my cards
	//compare their old hand with my old hand
	//compare their new hand with my new hand
	//if the winner changes then increase int changes by 1.

    if(roundNumber == 2)
    {
        commCards = 3;
        commCardsBefore = 0;
    }
    else if(roundNumber == 3)
    {
        commCards = 4;
        commCardsBefore = 3;
    }
    else if(roundNumber == 4)
    {
        commCards = 5;
        commCardsBefore = 4;
    }
    for(int j = 0; j < commCards; j ++)
    {
        existingCards[j] = communityCards[j];
        existingSuits[j] = communitySuits[j];
    }
    for(int i = 0; i < samples; i ++)
    {
        winBefore = 0;
        winAfter = 0;
        //generate remaining community cards
        for(int j = commCards; j < 5; j ++)
        {
            dealCard(existingCards, existingSuits, deal);
            existingCards[j] = deal[0];
            existingSuits[j] = deal[1];
            communityCards[j] = deal[0];
            communitySuits[j] = deal[1];
        }
        //generate opponent's cards
        dealCard(existingCards, existingSuits, deal);
        oppCards[0] = deal[0];
        oppSuits[0] = deal[1];
        existingCards[5] = deal[0];
        existingSuits[5] = deal[1];
        dealCard(existingCards, existingSuits, deal);
        oppCards[1] = deal[0];
        oppSuits[1] = deal[1];
        existingCards[6] = deal[0];
        existingSuits[6] = deal[1];
        //fill in my and opponent's cards
        for(int k = 0; k < 5; k ++)
        {
            myCards[k + 2] = communityCards[k];
            mySuits[k + 2] = communitySuits[k];
            oppCards[k + 2] = communityCards[k];
            oppSuits[k + 2] = communitySuits[k];
        }
        myHandValue = getHandScore(myCards, mySuits);
        oppHandValue = getHandScore(oppCards, oppSuits);
        if(myHandValue > oppHandValue)
        {
            winAfter = 1;
        }
        //replace newest community card(s) with a random card to assess winning beforehand
        for(int j = commCardsBefore; j < commCards; j++)
        {
            existingCards[j] = 0;
            existingSuits[j] = 0;
        }
        for(int j = commCardsBefore; j < commCards; j++)
        {
            myCards[j + 2] = 0;
            mySuits[j + 2] = 0;
            oppCards[j + 2] = 0;
            oppSuits[j + 2] = 0;
        }
        myHandValue = getHandScore(myCards, mySuits);
        oppHandValue = getHandScore(oppCards, oppSuits);
        if(myHandValue > oppHandValue)
        {
            winBefore = 1;
        }
        if((winBefore - winAfter) != 0)
        {
            changes ++;
        }
    }
    averageChange = changes / samples;
    return averageChange;
}

float winProbChange(float communityCards[5], float communitySuits[5], int roundNumber)
{   //winProbChange finds how much the newest community card changes the strength of hole cards
    float strengthBefore, strengthAfter; //strength is the percent of hands which some hole cards beat. This is before and after the newest card
    float sumAbsChanges = 0; //sum of the absolute change in hand strength
    double sumSqChanges = 0, stDevChanges; //sumsqchanges used to calculate variance of changes
    float samples = 100; //made a float so that division can be done without truncation
    float averageChange;
    int commCards; //number of community cards present
    int commCardsBefore;
    float existingCards[9] = {0,0,0,0,0,0,0,0,0};
    float existingSuits[9] = {0,0,0,0,0,0,0,0,0};
    float holeCards[2] = {0,0};
    float holeSuits[2] = {0,0};
    float priorCommCards[5] = {0,0,0,0,0};
    float priorCommSuits[5] = {0,0,0,0,0};
    float deal[2];
	//generate hole cards, assess change in winning probability before and after the newest community card

    if(roundNumber == 2)
    {
        commCards = 3;
        commCardsBefore = 0;
    }
    else if(roundNumber == 3)
    {
        commCards = 4;
        commCardsBefore = 3;
    }
    else if(roundNumber == 4)
    {
        commCards = 5;
        commCardsBefore = 4;
    }
    for(int i = 0; i < samples; i++)
    {
        for(int j = 0; j < commCards; j++)
        {
            existingCards[j] = communityCards[j];
            existingSuits[j] = communitySuits[j];
            if(j < commCardsBefore)
            {
                priorCommCards[j] = communityCards[j];
                priorCommSuits[j] = communitySuits[j];
            }
            else
            {
                priorCommCards[j] = 0;
                priorCommSuits[j] = 0;
            }
            if(j >= commCards)
            {
                communityCards[j] = 0;
                communitySuits[j] = 0;
            }
        }
        //deal hole cards
        for(int j = 0; j < 2; j++)
        {
            dealCard(existingCards, existingSuits, deal);
            holeCards[j] = deal[0];
            holeSuits[j] = deal[1];
            existingCards[commCards] = deal[0];
            existingSuits[commCards] = deal[1];
        }

        //calculate probability of winning before and after
        strengthBefore = winProb(holeCards, holeSuits, priorCommCards, priorCommSuits, 2);
        strengthAfter = winProb(holeCards, holeSuits, communityCards, communitySuits, 2);
        if(strengthAfter > strengthBefore)
        {
            sumAbsChanges = sumAbsChanges + strengthAfter - strengthBefore;
        }
        else
        {
            sumAbsChanges = sumAbsChanges + strengthBefore - strengthAfter;
        }
        sumSqChanges = sumSqChanges + pow((strengthAfter - strengthBefore), 2);
    }
    averageChange = sumAbsChanges / samples;
    stDevChanges = pow((sumSqChanges / samples - pow(averageChange, 2)), 0.5);
    cout << "averagechange is " << averageChange << endl;
    cout << "stdevchanges is " << stDevChanges << endl;
    return averageChange;
}

int oneLayerFeedForward(double currentLayerNodes[], int currentLayerSize, double nextLayerNodes[], int nextLayerSize, double weights[maxLayerSize][maxLayerSize], int applySigmoid)
{   //oneLayerFeedForward calculates one layer of the neural network with the sigmoid function applied
    //nextlayernodes is modified to give the output values without an array returned
    //applysigmoid indicates if the sigmoid function is applied to the output

    currentLayerNodes[0] = 1; //first neuron in a layer is the bias. Previous steps to calculate the first neuron's output are redundant but this does not use much computing time
    //fill output array with figures before sigmoid function is applied
    for(int i = 0; i < nextLayerSize; i ++)
    {
        for(int j = 0; j < currentLayerSize; j ++)
        {
            nextLayerNodes[i] += currentLayerNodes[j] * weights[j][i];
        }
    }
    if(applySigmoid == 1)
    {
        //apply sigmoid function
        for(int i = 0; i < nextLayerSize; i ++)
        {
            nextLayerNodes[i] = 1 / (1 + pow(2.718282, -1 * nextLayerNodes[i]));
        }
    }
    return 0;
}

double normalizeUniformVariable(double variable, double mean, double range)
{   //normalize a uniformly distributed variable for the neural network inputs
    //normalized variable is in the range [-1,1]
    double normalizedVariable = 2 * ((variable - mean) / range);
    return normalizedVariable;
}

double normalizeExponentialVariable(double variable, double rate, double minimum)
{   //normalize an exponentially distributed variable for the neural network inputs
    //normalized variable is in the range [-1,1]
    double normalizedVariable = 1 - 2 * pow(2.71828, -1 * rate * (minimum + variable));
    return normalizedVariable;
}

double normalizeNormalVariable(double variable, double mean, double stDev)
{   //normalize a normally distributed variable for the neural network inputs
    //normalized variable has mean 0 and stdev 1
    double normalizedVariable = ((variable - mean) / stDev);
    return normalizedVariable;
}

int linearScaleInputWeights(int inputPosition, double scaleConstant, int layerSizes[numberLayers])
{   //linearScaleInputWeights modifies multiplies input weights by a constant to scale them
    for(int familyNumber = 0; familyNumber < familyCount; familyNumber ++)
    {
        for(int memberNumber = 0; memberNumber < familyMembers; memberNumber ++)
        {
            //open file
            stringstream ss1;
            ss1 << familyNumber;
            string chromosomeFamilyString = ss1.str();
            stringstream ss2;
            ss2 << memberNumber;
            string chromosomeMemberString = ss2.str();
            string chromosomeFileName = "family";
            chromosomeFileName.append(chromosomeFamilyString);
            chromosomeFileName.append("member");
            chromosomeFileName.append(chromosomeMemberString);
            chromosomeFileName.append(".txt");
            ifstream chromosomeWeightsFile( chromosomeFileName.c_str() );

            double playerWeights[numberLayers][inputLayerSize][hiddenLayerSize];
            //read through each layer and store in array chromosomeWeights
            for(int i = 0; i < (numberLayers - 1); i ++)
            {
                for(int j = 0; j < layerSizes[i]; j ++)
                {
                    for(int k = 0; k < layerSizes[i + 1]; k ++)
                    {
                        chromosomeWeightsFile >> playerWeights[i][j][k];
                    }
                }
            }
            chromosomeWeightsFile.close();

            //rescale input weights
            for(int i = 0; i < layerSizes[1]; i ++)
            {
                //multiply input being modified by a constant
                playerWeights[0][inputPosition][i] = playerWeights[0][inputPosition][i] * scaleConstant;
            }

            ofstream chromosomeWeightsFile2( chromosomeFileName.c_str() );
            //write new weights to chromosomeWeightsFile2

            for(int i = 0; i < (numberLayers - 1); i ++)
            {
                for(int j = 0; j < layerSizes[i]; j ++)
                {
                    for(int k = 0; k < layerSizes[i + 1]; k ++)
                    {
                        chromosomeWeightsFile2 << playerWeights[i][j][k] << "\t";
                    }
                    chromosomeWeightsFile2 << "\n";
                }
            }
        }
    }
    return 0;
}

int rescaleInputWeights(int inputPosition, int layerSizes[numberLayers], double oldMean, double oldRange, double newMean, double newRange, int familyNumber, int memberNumber)
{   //rescaleInputWeights modifies weights when the mean and range/stdev changes so that it produces the same results as before
    //Range is replaced with standard deviation if variable is uniformly distributed

    //open file
    stringstream ss1;
    ss1 << familyNumber;
    string chromosomeFamilyString = ss1.str();
    stringstream ss2;
    ss2 << memberNumber;
    string chromosomeMemberString = ss2.str();
    string chromosomeFileName = "family";
    chromosomeFileName.append(chromosomeFamilyString);
    chromosomeFileName.append("member");
    chromosomeFileName.append(chromosomeMemberString);
    chromosomeFileName.append(".txt");
    ifstream chromosomeWeightsFile( chromosomeFileName.c_str() );

    double playerWeights[numberLayers][inputLayerSize][hiddenLayerSize];
    //read through each layer and store in array chromosomeWeights
    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        for(int j = 0; j < layerSizes[i]; j ++)
        {
            for(int k = 0; k < layerSizes[i + 1]; k ++)
            {
                chromosomeWeightsFile >> playerWeights[i][j][k];
            }
        }
    }
    chromosomeWeightsFile.close();

    //rescale input weights
    for(int i = 0; i < layerSizes[1]; i ++)
    {
        //rescale bias input
        //for all weights coming out of bias
        //newBias = oldBias - oldInputWeight * (2 * (newMean - oldMean) / oldRange - 0.5);
        playerWeights[0][0][i] = playerWeights[0][0][i] - 2 * playerWeights[0][inputPosition][i] * (oldMean - newMean) / oldRange;

        //rescale input being modified
        //for all weights coming from inputPosition
        //newInputWeight = (oldInputWeight * newRange) / oldrange;
        playerWeights[0][inputPosition][i] = playerWeights[0][inputPosition][i] * (newRange / oldRange);
    }

    ofstream chromosomeWeightsFile2( chromosomeFileName.c_str() );
    //write modified weights to chromosomeWeightsFile2

    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        for(int j = 0; j < layerSizes[i]; j ++)
        {
            for(int k = 0; k < layerSizes[i + 1]; k ++)
            {
                chromosomeWeightsFile2 << playerWeights[i][j][k] << "\t";
            }
            chromosomeWeightsFile2 << "\n";
        }
    }

    return 0;
}

int rescaleAllChromosomeInputWeights(int inputPosition, int layerSizes[numberLayers], double oldMean, double oldRange, double newMean, double newRange)
{   //rescaleAllChromosomeInputWeights modifies weights when the mean and range/stdev changes so that it produces the same results as before. This is done for every chromosome
    for(int i = 0; i < familyCount; i ++)
    {
        for(int j = 0; j < familyMembers; j ++)
        {
            //repeat rescaling for every member of every family
            rescaleInputWeights(inputPosition, layerSizes, oldMean, oldRange, newMean, newRange, i, j);
        }
    }
    return 0;
}

int neuralNetwork(double pot, double handStrength, double callValue, double existingBet, double roundNumber, double playersActive, double initialNumberPlayers, double bigBlind, double initialChipsAverage, int raises[maxPlayers], int folds[maxPlayers], double weights01[maxLayerSize][maxLayerSize], double weights12[maxLayerSize][maxLayerSize])
{   //NeuralNetwork takes inputs and weights and returns the amount to bet
    //weights01 is the weights for the connections between the 0th and 1st layers of the neural
    //existingBet is the amount which the player has already staked

    double inputLayer[inputLayerSize] = {0};
    double hiddenLayer[hiddenLayerSize] = {0};
    double outputLayer[outputLayerSize] = {0};
    int amountBet;
    inputLayer[0] = 1; //bias input in neural network
    inputLayer[1] = normalizeUniformVariable(log(pot / bigBlind), logPotMean, logPotRange);
    inputLayer[2] = normalizeUniformVariable(handStrength, handStrengthMean, handStrengthRange);
    if(callValue == 0)
    {
        inputLayer[3] = -3; //normalized callValue is set to -3 to distinguish between very low call and no call
    }
    else
    {
        inputLayer[3] = normalizeUniformVariable(log(callValue / bigBlind), logCallValueMean, logCallValueRange);
    }

    if(existingBet == 0)
    {
        inputLayer[4] = -1.5; //normalized existingBet is set to -1.5 to distinguish between very low existing bet and no existing bet
    }
    else
    {
        inputLayer[4] = normalizeUniformVariable(log(1 + (existingBet / bigBlind)), logExistingBetMean, logExistingBetRange);
    }

    inputLayer[5] = normalizeUniformVariable(roundNumber, roundNumberMean, roundNumberRange);

    inputLayer[6] = normalizeNormalVariable(playersActive, playersActiveMean, playersActiveStDev);

    inputLayer[7] = normalizeUniformVariable(initialNumberPlayers, initialNumberPlayersMean, initialNumberPlayersRange);

    //sort raises to set as input variables
    int orderedRaises[maxPlayers];
    for(int i = 0; i < maxPlayers; i ++)
    {
        if(!folds[i])
        {
            orderedRaises[i] = raises[i];
        }
        else
        {//if player is folded set their raises to 0 since they are not a threat
            orderedRaises[i] = 0;
        }
    }
    sort(orderedRaises, orderedRaises + maxPlayers, std::greater<int>());
    inputLayer[8] = normalizeUniformVariable(log(1 + (orderedRaises[0] / bigBlind)), logFirstTopRaiseMean , logFirstTopRaiseRange);
    inputLayer[9] = 0;
    inputLayer[10] = 0;
    ///inputLayer[9] = normalizeUniformVariable(log(1 + (orderedRaises[1] / bigBlind)), logSecondTopRaiseMean , logSecondTopRaiseRange);
    ///inputLayer[10] = normalizeUniformVariable(log(1 + (orderedRaises[2] / bigBlind)), logThirdTopRaiseMean , logThirdTopRaiseRange);

    inputLayer[11] = normalizeUniformVariable(log(initialChipsAverage / bigBlind), logOpponentsInitialChipsMean , logOpponentsInitialChipsRange);



    //record input values for later analysis

    fstream potFile;
    potFile.open ("potRecords.txt", fstream::in | fstream::out | fstream::app);
    potFile << inputLayer[1] << endl;

    fstream handStrengthFile;
    handStrengthFile.open ("handStrengthRecords.txt", fstream::in | fstream::out | fstream::app);
    handStrengthFile << inputLayer[2] << endl;

    fstream callValueFile;
    callValueFile.open ("callValueRecords.txt", fstream::in | fstream::out | fstream::app);
    callValueFile << inputLayer[3] << endl;

    fstream existingBetFile;
    existingBetFile.open ("existingBetValueRecords.txt", fstream::in | fstream::out | fstream::app);
    existingBetFile << inputLayer[4] << endl;

    fstream roundNumberFile;
    roundNumberFile.open ("roundNumberRecords.txt", fstream::in | fstream::out | fstream::app);
    roundNumberFile << inputLayer[5] << endl;

    fstream playersActiveFile;
    playersActiveFile.open ("playersActiveRecords.txt", fstream::in | fstream::out | fstream::app);
    playersActiveFile << inputLayer[6] << endl;

    fstream initialPlayersActiveFile;
    initialPlayersActiveFile.open ("InitialPlayersActiveRecords.txt", fstream::in | fstream::out | fstream::app);
    initialPlayersActiveFile << inputLayer[7] << endl;

    fstream topRaiseFile;
    topRaiseFile.open ("topRaiseActiveRecords.txt", fstream::in | fstream::out | fstream::app);
    topRaiseFile << inputLayer[8] << endl;

    fstream raisesFile;
    raisesFile.open ("raisesRecords.txt", fstream::in | fstream::out | fstream::app);
    raisesFile << orderedRaises[0] << "\t" << orderedRaises[1] <<"\t" << orderedRaises[2] <<"\t" << orderedRaises[3] <<"\t" << orderedRaises[4] <<"\t" << orderedRaises[5] <<"\t" << orderedRaises[6] << endl;

    fstream averageChipsFile;
    averageChipsFile.open ("AvgChipsRecords.txt", fstream::in | fstream::out | fstream::app);
    averageChipsFile <<  inputLayer[11] << endl;


    //put input variables through neural network algorithm

    oneLayerFeedForward(inputLayer, inputLayerSize, hiddenLayer, hiddenLayerSize, weights01, 1);

    oneLayerFeedForward(hiddenLayer, hiddenLayerSize, outputLayer, outputLayerSize, weights12, 1);

    //if the first output is less than 0.5 (0 without sigmoid) then check/fold
    if(outputLayer[0] < 0.5)
    {
        amountBet = 0;
    }
    //else if the second output is less than 0.5 (0 without sigmoid) then call
    else if(outputLayer[1] < 0.5)
    {
        amountBet = callValue;
    }
    //else raise
    else
    {
        amountBet = callValue + (outputLayer[2] * initialChipsAverage); //amount raised is between 0 and the average initial chips stack of opponents
        ///amountBet = callValue + ((((2 * outputLayer[1]) - 1) * pot) + (((2 * outputLayer[2]) - 1) * initialChipsAverage);
    }

    if(amountBet < 0)
    {
        amountBet = 0;
    }

    return amountBet;
}

int simpleDecision(int position, int callValue, int chips, int pot, int bigBlind, int calls[maxPlayers], int raises[maxPlayers], double handStrength, int playersActive)
{   //simpleDecision returns a bet based on hand strength only
    float winChance = handStrength; //chance of winning against one other player
    int newBet;
    if(winChance < 0.1)
    {
        newBet = 0;
    }
    else if(winChance < 0.3)
    {
        newBet = callValue;
    }
    else
    {
        newBet = callValue + pot * 0.1;
    }
    if(newBet > chips)
    {
        newBet = 0;
    }
    return newBet;
}

int decision(int position, int callValue, int chips[maxPlayers], int pot, int bigBlind, int calls[maxPlayers], int raises[maxPlayers], int folds[maxPlayers], double handStrength, int roundNumber, int playersActive, int initialNumberPlayers, double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize])
{   //decision returns bet which is got using neural network decision method
    double winChance = handStrength; //winChance is the probability of beating one player's cards, not all players
    int myChips = chips[position];
    int newBet;

    //find amount already bet
    int existingBet = calls[position] + raises[position];

    //calculate the average initial chip stack for remaining opponents
    float initialChipsCount = 0;
    double initialChipsTotal = 0;
    int initialChipsAverage;
    for(int i = 0; i < maxPlayers; i ++)
    {
        if((!folds[i]) && (i != position))
        {
            initialChipsCount ++;
            initialChipsTotal += chips[i] + calls[i] + raises[i];
        }
    }
    initialChipsAverage = (initialChipsTotal / initialChipsCount);

    //fill weight arrays for individual layers
    double weights01[maxLayerSize][maxLayerSize];
    double weights12[maxLayerSize][maxLayerSize];
    for(int i = 0; i < maxLayerSize; i ++)
    {
        for(int j = 0; j < maxLayerSize; j ++)
        {
            weights01[i][j] = playerWeights[position][0][i][j];
            weights12[i][j] = playerWeights[position][1][i][j];
        }
    }

    newBet = neuralNetwork(pot, winChance, callValue, existingBet, roundNumber, playersActive, initialNumberPlayers, bigBlind, initialChipsAverage, raises, folds, weights01, weights12);

    if(newBet > myChips)
    {
        newBet = myChips;
    }

    fstream myChipsFile;
    myChipsFile.open ("myChipsRecords.txt", fstream::in | fstream::out | fstream::app);
    myChipsFile << (myChips / bigBlind) << endl;

    fstream newBetFile;
    newBetFile.open ("newBetRecords.txt", fstream::in | fstream::out | fstream::app);
    newBetFile << (newBet / bigBlind) << endl;

    return newBet;
}

int getBet(int maxBet, int position, int pot, int bigBlind, string playerNames[maxPlayers], int aiPlayers[maxPlayers], int chips[maxPlayers], int bets[maxPlayers], int calls[maxPlayers], int raises[maxPlayers], int folds[maxPlayers], double handStrength, int roundNumber, int playersActive, int initialNumberPlayers, double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize])
{   //return the new bet made by either AI or human players
    int newBet = -1;
    int callValue = maxBet - bets[position];
    if(callValue < 0)
    {
        callValue = 0;
    }
    //if it is an AI player then use decision algorithm
    if(aiPlayers[position] == 1)
    {
        //newBet = simpleDecision(position, callValue, chips[position], pot, bigBlind, calls, raises, handStrength, playersActive);
        newBet = decision(position, callValue, chips, pot, bigBlind, calls, raises, folds, handStrength, roundNumber, playersActive, initialNumberPlayers, playerWeights);
    }
    else
    {
        while(newBet < 0)
        {
            newBet = getHumanBet(position, maxBet, pot, playerNames, chips, bets, calls, raises);
            if(newBet < 0)
            {
                cout << "Enter a valid bet" << endl;
            }
        }
    }
    return newBet;
}

int updateValues(int trainingMode, int position, int newBet, int maxBet, int playersActive, int pot, int active[maxPlayers], int chips[maxPlayers], int calls[maxPlayers], int bets[maxPlayers], int raises[maxPlayers], int folds[maxPlayers], string playerNames[maxPlayers], int updatedValues[3])
{   //updateValues takes a valid new bet and the current state of the game and updates the state of the game. Integers values which aren't in an array are returned in updatedValues array "updatedValues"

    //if player folds
    if((newBet == 0) && (bets[position] < maxBet) && (chips[position] > 0) && (!folds[position]))
    {//conditions for folding: new bet of 0, having already bet less than the max bet, having some chips still to bet, and not be folded already
        if(!trainingMode)
        {
            cout << endl  << playerNames[position] << " has folded" << endl;
        }
        playersActive --;
        folds[position] = 1;
    }

    //if player checks
    if((newBet == 0) && ((maxBet == bets[position]) && (chips[position] > 0)))
    {
        if(!trainingMode)
        {
            cout << endl  << playerNames[position] << " has checked" << endl;
        }
    }

    //check if player can afford to call
    if((chips[position] + bets[position]) >= maxBet)
    {
        //if player raises or calls
        if(newBet >= (maxBet - bets[position]))
        {
            calls[position] += maxBet - bets[position];
            raises[position] += bets[position] + newBet - maxBet;
            maxBet = bets[position] + newBet;
        }
    }
    else
    {
        //if player goes all in
        if(newBet == chips[position])
        {
            calls[position] += newBet;
        }
    }

    bets[position] += newBet;
    chips[position] -= newBet;
    active[position] = 1;
    pot += newBet;

    //announce bet made
    if(newBet != 0)
    {
        if(!trainingMode)
        {
            cout << endl << playerNames[position] << " has bet " << newBet << ". " << playerNames[position] << " now has " << chips[position] << " chips" << endl;
        }
    }

    updatedValues[0] = playersActive;
    updatedValues[1] = maxBet;
    updatedValues[2] = pot;

    return 0;
}

int setBlinds(int dealerPosition, int bigBlind, int chips[maxPlayers], int bets[maxPlayers], int calls[maxPlayers], int playersKnockedOut[maxPlayers], int blindInfo[3])
{   //setBlinds finds the position of the small and big blinds and sets their bet. The positions of the blinds, the blind bets, and the pot are returned in an array
    int smallBlindPosition, bigBlindPosition;
    //find small blind position
    for(int i = (dealerPosition + 1); i < (dealerPosition + maxPlayers); i ++)
    {   //starting 1 seat after the dealer, search for the position of the small blind player - the first player after the dealer who is not knocked out
        if(!playersKnockedOut[i % maxPlayers])
        {
            smallBlindPosition = i % maxPlayers;
            i = (dealerPosition + maxPlayers); //stop loop
        }
    }

    //find big blind position
    for(int i = (smallBlindPosition + 1); i < (dealerPosition + maxPlayers + 1); i ++)
    {   //starting 1 seat after the small blind, search for the position of the big blind player - the first player not knocked out
        if(!playersKnockedOut[i % maxPlayers])
        {
            bigBlindPosition = i % maxPlayers;
            i = (dealerPosition + maxPlayers + 1);
        }
    }

    //calculate blind bets and update bets, chips & calls
    int smallBlindBet, bigBlindBet;

    //if small blind player cannot afford to bet small blind then go all in
    if(chips[smallBlindPosition] >= (bigBlind / 2))
    {
        smallBlindBet = bigBlind / 2;
    }
    else
    {
        smallBlindBet = chips[smallBlindPosition];
    }
    bets[smallBlindPosition] = smallBlindBet;
    chips[smallBlindPosition] -= smallBlindBet;
    calls[smallBlindPosition] = smallBlindBet;

    //if big blind player cannot afford to bet big blind then go all in
    if(chips[bigBlindPosition] >= bigBlind)
    {
        bigBlindBet = bigBlind;
    }
    else
    {
        bigBlindBet = chips[bigBlindPosition];
    }
    bets[bigBlindPosition] = bigBlindBet; //set small blind
    chips[bigBlindPosition] -= bigBlindBet;
    calls[bigBlindPosition] = bigBlindBet;

    int pot = bigBlindBet + smallBlindBet;

    blindInfo[0] = smallBlindPosition;
    blindInfo[1] = bigBlindPosition;
    blindInfo[2] = pot;

    return 0;
}

int giveWinnings(int chips[maxPlayers], int bets[maxPlayers], int folds[maxPlayers], string playerNames[maxPlayers], int trainingMode, float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float communityCards[5], float communitySuits[5])
{   //giveWinnings find the winner and gives them the chips won
    float holeCards[2], holeSuits[2];
    //print cards of players who are not folded
    if(!trainingMode)
    {
        for(int i = 0; i < maxPlayers; i ++)
        {
            if(!folds[i])
            {
                holeCards[0] = playerCards[i][0];
                holeCards[1] = playerCards[i][1];
                holeSuits[0] = playerSuits[i][0];
                holeSuits[1] = playerSuits[i][1];
                cout << playerNames[i] << "'s cards are " << endl;
                for(int k = 0; k < 2; k ++)
                {
                    char temp = getSuitLetter(holeSuits[k]);
                    if(holeCards[k] > 10)
                    {
                        char temp2 = getCardLetter(holeCards[k]);
                        cout << temp2 << " " << temp << endl;
                    }
                    else
                    {
                        int temp2 = holeCards[k];
                        cout << temp2 << " " << temp << endl;
                    }
                }
            }
        }
    }

    //find the winner

    int winnerPositions[maxPlayers];
    findWinners(winnerPositions, playerCards, playerSuits, communityCards, communitySuits, folds);

    //give chips to winner(s)
    int sumWinnings; //amount of chips won by a player
    for(int k = 0; k < maxPlayers; k ++)
    {   //loop through all winners
        if(!folds[winnerPositions[k]])
        {
            sumWinnings = 0;
            int winnerBet = bets[winnerPositions[k]];
            if(!trainingMode && (bets[winnerPositions[k]] != 0))
            {   //if the next winner has 0 bets remaining to claim profit on then do not announce them as a winner
                cout << "The ";
                if(k == 0)
                {
                    cout << "first ";
                }
                else
                {
                    cout << "next ";
                }
                cout << "winner is " << playerNames[winnerPositions[k]] << endl;
            }
            for(int j = 0; j < maxPlayers; j ++)
            {
                //loop through all players to pay winners
                if(winnerBet >= bets[j])
                {
                    sumWinnings += bets[j];
                    bets[j] = 0;
                }
                else
                {
                    sumWinnings += winnerBet;
                    bets[j] -= winnerBet;
                }
            }
            chips[winnerPositions[k]] += sumWinnings;
        }
    }
    return 0;
}

int storeHumanDecision(int position, int newBet, double bigBlind, int maxBet, int callValue, int playersActive, int initialNumberPlayers, int roundNumber, double handStrength, float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float communityCards[5], float communitySuits[5], int pot, int active[maxPlayers], int chips[maxPlayers], int calls[maxPlayers], int bets[maxPlayers], int raises[maxPlayers], int folds[maxPlayers])
{   //storeHumanDecision writes to a file the most recent bet made and the variables describing the state of the game
    fstream humanDecisionFile;
    humanDecisionFile.open ("humanDecisions.txt", fstream::in | fstream::out | fstream::app);

    //write new bet
    humanDecisionFile << (newBet / bigBlind) << "\t";

    //write position, pot, playersActive, initialNumberPlayers, maxBet, callValue, roundNumber, handStrength
    humanDecisionFile << position << "\t";
    humanDecisionFile << (pot / bigBlind) << "\t";
    humanDecisionFile << playersActive << "\t";
    humanDecisionFile << initialNumberPlayers << "\t";
    humanDecisionFile << maxBet << "\t";
    humanDecisionFile << callValue << "\t";
    humanDecisionFile << roundNumber << "\t";
    humanDecisionFile << handStrength << "\t";

    //write hole cards
    for(int i = 0; i < 2; i ++)
    {
        humanDecisionFile << playerCards[position][i] << "\t";
    }

    //write hole suits
    for(int i = 0; i < 2; i ++)
    {
        humanDecisionFile << playerSuits[position][i]  << "\t";
    }

    //write community cards
    for(int i = 0; i < 5; i ++)
    {
        humanDecisionFile << communityCards[i] << "\t";
    }

    //write community suits
    for(int i = 0; i < 5; i ++)
    {
        humanDecisionFile << communitySuits[i] << "\t";
    }

    //write folds
    for(int i = 0; i < maxPlayers; i ++)
    {
        humanDecisionFile << folds[i] << "\t";
    }

    //write chips
    for(int i = 0; i < maxPlayers; i ++)
    {
        humanDecisionFile << (chips[i] / bigBlind) << "\t";
    }

    //write bets
    for(int i = 0; i < maxPlayers; i ++)
    {
        humanDecisionFile << (bets[i] / bigBlind) << "\t";
    }

    //write calls
    for(int i = 0; i < maxPlayers; i ++)
    {
        humanDecisionFile << (calls[i] / bigBlind) << "\t";
    }

    //write raises
    for(int i = 0; i < maxPlayers; i ++)
    {
        humanDecisionFile << (raises[i] / bigBlind) << "\t";
    }

    return 0;
}

int storeAiDecision(int position, int newBet, double bigBlind, int maxBet, int callValue, int playersActive, int initialNumberPlayers, int roundNumber, double handStrength, float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float communityCards[5], float communitySuits[5], int pot, int active[maxPlayers], int chips[maxPlayers], int calls[maxPlayers], int bets[maxPlayers], int raises[maxPlayers], int folds[maxPlayers])
{   //storeAiDecision writes to a file the most recent bet made and the variables describing the state of the game
    fstream AiDecisionFile;
    AiDecisionFile.open ("AiDecisions.txt", fstream::in | fstream::out | fstream::app);

    //write new bet
    AiDecisionFile << (newBet / bigBlind) << "\t";

    //write position, pot, playersActive, initialNumberPlayers, maxBet, callValue, roundNumber, handStrength
    AiDecisionFile << position << "\t";
    AiDecisionFile << (pot / bigBlind) << "\t";
    AiDecisionFile << playersActive << "\t";
    AiDecisionFile << initialNumberPlayers << "\t";
    AiDecisionFile << maxBet << "\t";
    AiDecisionFile << callValue << "\t";
    AiDecisionFile << roundNumber << "\t";
    AiDecisionFile << handStrength << "\t";

    //write hole cards
    for(int i = 0; i < 2; i ++)
    {
        AiDecisionFile << playerCards[position][i] << "\t";
    }

    //write hole suits
    for(int i = 0; i < 2; i ++)
    {
        AiDecisionFile << playerSuits[position][i]  << "\t";
    }

    //write community cards
    for(int i = 0; i < 5; i ++)
    {
        AiDecisionFile << communityCards[i] << "\t";
    }

    //write community suits
    for(int i = 0; i < 5; i ++)
    {
        AiDecisionFile << communitySuits[i] << "\t";
    }

    //write folds
    for(int i = 0; i < maxPlayers; i ++)
    {
        AiDecisionFile << folds[i] << "\t";
    }

    //write chips
    for(int i = 0; i < maxPlayers; i ++)
    {
        AiDecisionFile << (chips[i] / bigBlind) << "\t";
    }

    //write bets
    for(int i = 0; i < maxPlayers; i ++)
    {
        AiDecisionFile << (bets[i] / bigBlind) << "\t";
    }

    //write calls
    for(int i = 0; i < maxPlayers; i ++)
    {
        AiDecisionFile << (calls[i] / bigBlind) << "\t";
    }

    //write raises
    for(int i = 0; i < maxPlayers; i ++)
    {
        AiDecisionFile << (raises[i] / bigBlind) << "\t";
    }

    return 0;
}

int storeDecision(int AIplayer, int position, int newBet, double bigBlind, int maxBet, int playersActive, int initialNumberPlayers, int roundNumber, double handStrength, float playerCards[maxPlayers][2], float playerSuits[maxPlayers][2], float communityCards[5], float communitySuits[5], int pot, int active[maxPlayers], int chips[maxPlayers], int calls[maxPlayers], int bets[maxPlayers], int raises[maxPlayers], int folds[maxPlayers])
{   //storeAiDecision writes to a file the most recent bet made and the variables describing the state of the game
    //information is stored in a different file depending on whether a human or AI played
    int callValue = maxBet - bets[position];
    if(callValue < 0)
    {
        callValue = 0;
    }
    if(AIplayer == 1)
    {
        storeAiDecision(position, newBet, bigBlind, maxBet, callValue, playersActive, initialNumberPlayers, roundNumber, handStrength, playerCards, playerSuits, communityCards, communitySuits, pot, active, chips, calls, bets, raises, folds);
    }
    else
    {
        storeHumanDecision(position, newBet, bigBlind, maxBet, callValue, playersActive, initialNumberPlayers, roundNumber, handStrength, playerCards, playerSuits, communityCards, communitySuits, pot, active, chips, calls, bets, raises, folds);
    }
}

int storeProfit(int position, int aiPlayers[maxPlayers], int chips[maxPlayers], int initialChips[maxPlayers], double bigBlind)
{   //storeProfit records the profit earned following an action earlier in the game
    if(aiPlayers[position])
    {
        fstream AiDecisionFile;
        AiDecisionFile.open ("AiDecisions.txt", fstream::in | fstream::out | fstream::app);

        //write new bet
        AiDecisionFile << ((chips[position] - initialChips[position]) / bigBlind) << endl;
    }
    else
    {
        fstream humanDecisionFile;
        humanDecisionFile.open ("humanDecisions.txt", fstream::in | fstream::out | fstream::app);

        //write new bet
        humanDecisionFile << ((chips[position] - initialChips[position]) / bigBlind) << endl;
    }
    return 0;
}

int selectMove(int initialNumberPlayers)
{   //selectMove picks a random move/action based on the expected number of moves that are made in a game
    //number of actions made in a game is approx 2.5 for every player and
    int moveNumber = -1;
    int meanMoves, stDevMoves;

    meanMoves = 4 + 1.4 * initialNumberPlayers;
    stDevMoves = 4.7;
    std::normal_distribution<double> normaldistribution(meanMoves, stDevMoves);

    while(moveNumber < 0)
    {
        moveNumber = (normaldistribution(generator) + 0.5);
    }

    return moveNumber;
}

int playHand(int dealerPosition, int trainingMode, int aiPlayers[maxPlayers], int chips[maxPlayers], string playerNames[maxPlayers], int playersKnockedOut[maxPlayers], int numberPlayers, int bigBlind, int manualDealing, double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize])
{   //playhand plays one hand of poker and modifies players' chip counts
    //If trainingMode is true then nothing is printed and there are no "enter anything to continue" prompts

    int pot = 0;
    int playersActive = numberPlayers; //playersactive is number of players not folded, numberplayers is number of players not knocked out
    int initialNumberPlayers = numberPlayers;
    int position;
    int roundActive = 1; //is the round still being played
    int maxBet = bigBlind;
    int newBet;
    int roundNumber;
    int bets[maxPlayers] = {0}, calls[maxPlayers] = {0}, raises[maxPlayers] = {0}; //amount a player has bet/called/raised
    int folds[maxPlayers] = {0}; //0 if player hasn't folded, 1 otherwise
    int active[maxPlayers] = {0}; //0 if a player has not acted this betting round, 1 otherwise
    float playerCards[maxPlayers][2];
    float playerSuits[maxPlayers][2];
    float existingCards[5 + (2 * maxPlayers)] = {0};
    float existingSuits[5 + (2 * maxPlayers)] = {0};
    float communityCards[5] = {0}; //value of the cards which every player can use
    float communitySuits[5] = {0}; //value of the suits which every player can use
    float holeCards[2], holeSuits[2]; //cards and suits of a given player
    double handStrengths[maxPlayers];

    //variables used in recording training data
    int recordMoveNumber, positionStored, initialChips[maxPlayers], actionCount = 0; //actionCount is a running count of how many betting actions have been made in this game

    recordMoveNumber = selectMove(initialNumberPlayers);

    //do stuff that is needed before round 1 starts
    for(int k = 0; k < maxPlayers; k ++)
    {   //populate various arrays defining each player's situation
        playerCards[k][0] = 0;
        playerCards[k][1] = 0;
        playerSuits[k][0] = 0;
        playerSuits[k][1] = 0;
        if(playersKnockedOut[k])
        {
            folds[k] = 1;
        }
        initialChips[k] = chips[k];
    }

    int blindInfo[3];
    //set the blinds and starting position
    setBlinds(dealerPosition, bigBlind, chips, bets, calls, playersKnockedOut, blindInfo);
    int smallBlindPosition = blindInfo[0];
    int bigBlindPosition = blindInfo[1];
    pot = blindInfo[2];
    position = (bigBlindPosition + 1) % maxPlayers; //betting starts from 1 after the big blind if it is the first round

    //begin play for the four rounds of betting
    for(roundNumber = 1; roundNumber < 5; roundNumber ++)
    {
        //reset activity at the start of a round
        roundActive = 1;
        for(int k = 0; k < maxPlayers; k ++)
        {
            active[k] = 0;
        }

        //round 1-4 begins
        deal(roundNumber, dealerPosition, manualDealing, numberPlayers, trainingMode, aiPlayers, playersKnockedOut, playerNames, folds, playerCards, playerSuits, existingCards, existingSuits, communityCards, communitySuits);

        //print community cards
        if(!trainingMode)
        {
            if(roundNumber != 1)
            {
                cout << endl << "community cards are:" << endl;
                for(int k = 0; k < 5; k ++)
                {
                    if(communityCards[k] != 0)
                    {
                        char temp = getSuitLetter(communitySuits[k]);
                        if(communityCards[k] > 10)
                        {
                            char temp2 = getCardLetter(communityCards[k]);
                            cout << temp2 << " " << temp << endl;
                        }
                        else
                        {
                            int temp3 = communityCards[k];
                            cout << temp3 << " " << temp << endl;
                        }
                    }
                }
            }
        }

        //calculate players' hand strengths (probability of beating one other player)
        getHandStrengths(handStrengths, roundNumber, folds, playerCards, playerSuits, communityCards, communitySuits);

        //betting begins
        while(roundActive)
        {
            if(!folds[position] && !trainingMode)
            {//only make a line break when players aren't folded, this avoids inconsistent numbers of line breaks
                cout << endl;
            }
            newBet = 0;

            //check if round is over. Round ends if there is 1 player or if a player who has acted already has nothing to call
            if((playersActive == 1) || (active[position] && (bets[position] == maxBet)))
            {
                roundActive = 0;
                position = (position + maxPlayers - 1) % maxPlayers; //position must be decreased as the current position is the player who is already active
            }
            else
            {
                //check if this player has not folded (players are folded by default if knocked out)
                if(!folds[position]){
                    //check if this player cannot bet
                    if(chips[position] == 0)
                    {
                        if(!trainingMode)
                        {
                            cout << playerNames[position] << " cannot bet" << endl;
                        }
                    }
                    else
                    {
                        double handStrength = handStrengths[position];
                        newBet = getBet(maxBet, position, pot, bigBlind, playerNames, aiPlayers, chips, bets, calls, raises, folds, handStrength, roundNumber, playersActive, initialNumberPlayers, playerWeights);
                        actionCount ++;
                        if(actionCount == recordMoveNumber)
                        {   //record the state of the game and the new bet
                            storeDecision(aiPlayers[position], position, newBet, bigBlind, maxBet, playersActive, initialNumberPlayers, roundNumber, handStrength, playerCards, playerSuits, communityCards, communitySuits, pot, active, chips, calls, bets, raises, folds);
                            positionStored = position;
                        }
                    }
                    if(!trainingMode && aiPlayers[position])
                    {
                        cout << "Enter anything to continue" << endl;
                        string temp;
                        cin >> temp;
                        cout << endl;
                    }
                }
            }

            //update values after bet is made. playersActive, maxBet and pot are stored in the updatedInfo[3] array
            int updatedInfo[3];

            updateValues(trainingMode, position, newBet, maxBet, playersActive, pot, active, chips, calls, bets, raises, folds, playerNames, updatedInfo);
            playersActive = updatedInfo[0];
            maxBet = updatedInfo[1];
            pot = updatedInfo[2];
            position = (position + 1) % maxPlayers;
        } //while(roundactive) loop end

        //Find player who bet the most, if nobody matched their bet then reduce it to the second highest bet
        int tempBets[maxPlayers], betPositions[maxPlayers];
        for(int k = 0; k < maxPlayers; k ++)
        {
            tempBets[k] = bets[k];
            betPositions[k] = k;
        }
        sortBetters(tempBets, betPositions);
        if((bets[betPositions[0]] > bets[betPositions[1]]) && (bets[betPositions[0]] > bigBlind))
        {
            chips[betPositions[0]] += (bets[betPositions[0]] - bets[betPositions[1]]);
            raises[betPositions[0]] -= (bets[betPositions[0]] - bets[betPositions[1]]);
            bets[betPositions[0]] = bets[betPositions[1]];
            maxBet = bets[betPositions[1]];
            pot -= (bets[betPositions[0]] - bets[betPositions[1]]);
        }
    }
    //Show cards of those not folded and give chips to the winner(s)
    giveWinnings(chips, bets, folds, playerNames, trainingMode, playerCards, playerSuits, communityCards, communitySuits);

    //save the profit earned for the training data
    if(actionCount >= recordMoveNumber)
    {
        storeProfit(positionStored, aiPlayers, chips, initialChips, bigBlind);
    }

    for(int k = 0; k < maxPlayers; k ++)
    {
        if(!playersKnockedOut[k] && !trainingMode)
        {
            cout << playerNames[k] << " has a chip stack of " << chips[k] << endl;
        }
    }

    return 0;
}

int playManyHands(int bigBlind, int manualDealing, int trainingMode, int maxNumberHands, int initialPosition, string playerNames[maxPlayers], int aiPlayers[maxPlayers], int chips[maxPlayers], int playersKnockedOut[maxPlayers], double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize])
{   //playManyHands plays a game of poker until maxNumberHands have been played or until there is one player remaining. The players' chips are returned
    //maxNumberHands is the number of hands played before stopping. This is set to 0 if game continues until 1 player is left
    //initialPosition is the position of the dealer in the first game
    int dealerPosition;
    int gameActive = 1;
    int numberPlayers = countPlayers(playersKnockedOut);
    int handsPlayed = 0;
    int k = initialPosition;
    //play until a winner is found
    while(gameActive)
    {
        if(!playersKnockedOut[k % maxPlayers])
        {
            dealerPosition = k % numberPlayers;

            playHand(dealerPosition, trainingMode, aiPlayers, chips, playerNames, playersKnockedOut, numberPlayers, bigBlind, manualDealing, playerWeights);
            handsPlayed ++;
            cout << "number of hands played is " << handsPlayed << endl;
            cout << "number of players is " << numberPlayers << endl;
            if(!trainingMode)
            {
                cout << "enter anything to start next hand \n";
                string temp;
                cin >> temp;
            }
            for(int j = 0; j < maxPlayers; j ++)
            {   //update players knocked out
                if(chips[j] == 0)
                {
                    playersKnockedOut[j] = 1;
                }
            }
            numberPlayers = countPlayers(playersKnockedOut);
            if((numberPlayers == 1) || ((handsPlayed == maxNumberHands) && (maxNumberHands != 0)))
            {   //game ends if there is 1 player left or if the maximum number of hands has been played
                gameActive = 0;
            }
        }
        k ++;
    }
    return 0;
}

int selectPlayers(int numberPlayersHand, int gamesPlayed[], int playerRefNumbers[])
{   //selectPlayers modifies the playerRefNumbers array to fill it with those players who are to play the next hand and returns the minimum number of games which all players have played
    //half of the players chosen are those who have played the fewest games, this ensures that eventually all players have played enough games
    //numberPlayersHand is the number of players who are playing together
    //gamesPlayed is an array with the number of times each player has played
    //playerRefNumbers is an array with the reference numbers of each player who is playing this hand, this is modified
    //find the player who has played the fewest games and add them to playerRefNumber
    int minGames = gamesPlayed[0];
    int minGamesPlayer = 0; //the player with the fewest games played (initial guess: the player with Ref number 0
    for(int i = 1; i < (familyCount * familyMembers); i ++)
    {
        if(gamesPlayed[i] < minGames)
        {
            minGames = gamesPlayed[i];
            minGamesPlayer = i;
        }
    }
    playerRefNumbers[0] = minGamesPlayer;

    //select the remainder of the players to play in this hand at random
    std::uniform_int_distribution<int> uniformRefs(0, (familyCount * familyMembers) - 1);
    for(int i = 1; i < numberPlayersHand; i ++)
    {//pick a player at random, if they have not already been chosen then add them to the list of playerNumbers
        int refNumber = uniformRefs(generator);
        int uniqueRefNumber = 1;
        for(int j = 0; j < numberPlayersHand; j ++)
        {
            if(refNumber == playerRefNumbers[j])
            {//if true the reference number was not unique and a new one must be generated
                uniqueRefNumber = 0;
                j = numberPlayersHand;
                i --; //reduce i to repeat the selection for this player
            }
        }
        if(uniqueRefNumber == 1)
        {//if unique add the ref number to the list
            playerRefNumbers[i] = refNumber;
        }
    }

    //update games played by each player
    for(int i = 0; i < numberPlayersHand; i ++)
    {
        gamesPlayed[playerRefNumbers[i]] ++;
    }
    return minGames;
}

int generateChips(int bigBlind, float minChips, float maxChips, int numberPlayers, int chipStacks[maxPlayers])
{   //generateChips creates a log-uniform distributed variable between (minChips * bigBlind) and (maxChips * bigBlind)
    //minChips and maxChips are per-big blind
    int chipValue;
    double randomNumber, logRandomNumber;
    std::uniform_real_distribution<double> uniformdistribution(0.0,1.0);
    for(int i = 0; i < numberPlayers; i ++)
    {
        randomNumber = uniformdistribution(generator);
        logRandomNumber = log(minChips) + (randomNumber * (log(maxChips) - log(minChips)));
        chipValue = bigBlind * pow(2.71828, logRandomNumber);
        chipStacks[i] = chipValue;
    }

    //for those players not playing set their chipstack to 0
    for(int i = numberPlayers; i < maxPlayers; i ++)
    {
        chipStacks[i] = 0;
    }

    return 0;
}

int setToZero(double gameStats[familyCount * familyMembers][3])
{   //setToZero makes the gameStats matrix initially full of zeroes
    for(int i = 0; i < (familyCount * familyMembers); i ++)
    {
        for(int j = 0; j < 3; j ++)
        {
            gameStats[i][j] = 0;
        }
    }
    return 0;
}

int populateWeightsArray(int playerPosition, int refNumber, int layerSizes[numberLayers], double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize])
{   //populateWeightsArray fills the array containing neural network weights with the figures contained in a file, this is done for one player's weights
    int memberNumber = refNumber % familyMembers;
    int familyNumber = refNumber / familyMembers;
    stringstream ss1;
    ss1 << familyNumber;
    string familyString = ss1.str();
    stringstream ss2;
    ss2 << memberNumber;
    string memberString = ss2.str();
    string playerFileName = "family";
    playerFileName.append(familyString);
    playerFileName.append("member");
    playerFileName.append(memberString);
    playerFileName.append(".txt");

    ifstream playerWeightsFile( playerFileName.c_str() );
    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        for(int j = 0; j < layerSizes[i]; j ++)
        {
            for(int k = 0; k < layerSizes[i + 1]; k ++)
            {
                playerWeightsFile >> playerWeights[playerPosition][i][j][k];
            }
        }
    }
    playerWeightsFile.close();

    return 0;
}

int createGeneFiles(int layerSizes[numberLayers])
{   //createGeneFiles fills weights files with random numbers
    ///For each family and player create their file and initialise the NN weights
    ///For each player in the hand open their file and import the NN weights into a matrix playerWeights
    std::normal_distribution<double> normaldistribution(0.0, 1.0);
    for(int i = 0; i < familyCount; i ++)
    {
        for(int j = 0; j < familyMembers; j ++)
        {
            //create a string for the file name indicating the gene's family number
            stringstream ss1;
            ss1 << i;
            string familyString = ss1.str();
            stringstream ss2;
            ss2 << j;
            string memberString = ss2.str();
            string playerFileName = "family";
            playerFileName.append(familyString);
            playerFileName.append("member");
            playerFileName.append(memberString);
            playerFileName.append(".txt");
            ofstream playerWeightsFile( playerFileName.c_str() );
            for(int k = 0; k < (numberLayers - 1); k ++)
            {
                double doubleLayerSize = layerSizes[k];
                for(int l = 0; l < layerSizes[k]; l ++)
                {
                    for(int m = 0; m < layerSizes[k + 1]; m ++)
                    {
                        playerWeightsFile << (normaldistribution(generator))/ sqrt(doubleLayerSize) << "\t" ;
                    }
                    playerWeightsFile << "\n";
                }
            }
        }
    }
    return 0;
}

int modifyLayerSizes(int newLayerSizes[], int oldLayerSizes[], int oldNumberLayers, int familyNumber, int memberNumber)
{   //modifyLayerSizes introduces weights for new neurons in the neural network for one chromosome. New weights have a standard normal distribution
    //global variables maxLayerSize and NumberLayers must be set to the new layer parameters
    std::normal_distribution<double> normaldistribution(0.0, 1.0);

    double chromosomeWeights[numberLayers][maxLayerSize][maxLayerSize];

    //create array containing the old layer sizes and a size of 0 for any layers not existent in the old neural network
    int extendedOldLayerSizes[] = {0};
    for(int i = 0; i < numberLayers; i ++)
    {
        if(i < oldNumberLayers)
        {
            extendedOldLayerSizes[i] = oldLayerSizes[i];
        }
        else
        {
            extendedOldLayerSizes[i] = 0;
        }
    }

    //put old weights into chromosomeWeights array
    //open file
    stringstream ss1;
    ss1 << familyNumber;
    string chromosomeFamilyString = ss1.str();
    stringstream ss2;
    ss2 << memberNumber;
    string chromosomeMemberString = ss2.str();
    string chromosomeFileName = "family";
    chromosomeFileName.append(chromosomeFamilyString);
    chromosomeFileName.append("member");
    chromosomeFileName.append(chromosomeMemberString);
    chromosomeFileName.append(".txt");
    ifstream chromosomeWeightsFile( chromosomeFileName.c_str() );

    //read through layers and store in array chromosomeWeights
    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        for(int j = 0; j < oldLayerSizes[i]; j ++)
        {
            for(int k = 0; k < oldLayerSizes[i + 1]; k ++)
            {
                chromosomeWeightsFile >> chromosomeWeights[i][j][k];
            }
        }
    }

    chromosomeWeightsFile.close();

    //store random numbers in remaining elements of chromosomeWeights array
    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        double doubleLayerSize = newLayerSizes[i];
        if(oldLayerSizes[i] < newLayerSizes[i])
        {   //if this layer increased in size then create random weights from new nodes in this layer to all nodes in next layer
            for(int j = oldLayerSizes[i]; j < newLayerSizes[i]; j ++)
            {
                for(int k = 0; k < newLayerSizes[i + 1]; k ++)
                {
                    chromosomeWeights[i][j][k] = (normaldistribution(generator)) / sqrt(doubleLayerSize);
                }
            }
        }

        if(oldLayerSizes[i + 1] < newLayerSizes[i + 1])
        {   //if next layer increased in size then create random weights from all nodes in this layer to new nodes in next layer
            for(int j = 0; j < newLayerSizes[i]; j ++)
            {
                for(int k = oldLayerSizes[i + 1]; k < newLayerSizes[i + 1]; k ++)
                {
                    chromosomeWeights[i][j][k] = (normaldistribution(generator)) / sqrt(doubleLayerSize);
                }
            }
        }

        for(int j = oldLayerSizes[i]; j < newLayerSizes[i]; j ++)
        {
            for(int k = 0; k < newLayerSizes[i + 1]; k ++)
            {
                chromosomeWeights[i][j][k] = (normaldistribution(generator)) / sqrt(doubleLayerSize);
            }
        }
    }

    //store new array in file, overwrite old file.
    ofstream chromosomeWeightsFile2(chromosomeFileName.c_str() );

    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        for(int j = 0; j < newLayerSizes[i]; j ++)
        {
            for(int k = 0; k < newLayerSizes[i + 1]; k ++)
            {
                chromosomeWeightsFile2 << chromosomeWeights[i][j][k] << "\t";
            }
            chromosomeWeightsFile2 << "\n";
        }
    }
    return 0;
}

int addNewWeights(int newLayerSizes[], int oldLayerSizes[], int oldNumberLayers)
{   //addNewWeights introduces weights for new neurons in the neural network for all chromosomes
    //to use function set maxLayerSize and other globally defined layer sizes to the new sizes
    for(int i = 0; i < familyCount; i ++)
    {
        for(int j = 0; j < familyMembers; j ++)
        {
            modifyLayerSizes(newLayerSizes, oldLayerSizes, oldNumberLayers, i, j);
        }
    }
    return 0;
}

int setUpGame(int bigBlind, int maxChips, int minChips, int layerSizes[numberLayers], int playerRefNumbers[maxPlayers], int gamesPlayed[familyCount * familyMembers], int chips[maxPlayers], int aiPlayers[maxPlayers], string playerNames[maxPlayers], int playersKnockedOut[maxPlayers], double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize], int gameInfo[3])
{   //setUpGame chooses the players who will play the next game and assigned them an amount of chips. A dealer is also selected.
    //gameInfo stores the number of players, the dealer's position, and the minimum number of games which players have played
    int dealerPosition;
    int minGamesPlayed;
    //create random number generator for the number of players playing
    std::uniform_int_distribution<int> uniformPlayers(2, maxPlayers);

    //choose how many players will play
    int numberPlayersHand = uniformPlayers(generator);

    //select the players who will be playing, fill the playerRefNumbers array with those players
    minGamesPlayed = selectPlayers(numberPlayersHand, gamesPlayed, playerRefNumbers);

    //select how many chips each player will have, fill chips array
    generateChips(bigBlind, minChips, maxChips, numberPlayersHand, chips);

    //create random number generator for selecting which player is dealer
    std::uniform_int_distribution<int> uniformDealer(0, numberPlayersHand - 1);

    //select a dealer
    dealerPosition = uniformDealer (generator);

    //add the players' variables to the weights array
    for(int playerCount = 0; playerCount < numberPlayersHand; playerCount ++)
    {
        populateWeightsArray(playerCount, playerRefNumbers[playerCount], layerSizes, playerWeights);
    }

    //fill arrays required for playing one hand
    for(int i = 0; i < maxPlayers; i ++)
    {
        aiPlayers[i] = 1;
        playerNames[i] = " ";
        if(i < numberPlayersHand)
        {
            playersKnockedOut[i] = 0;
        }
        else
        {
            playersKnockedOut[i] = 1;
        }
    }

    //store information about the game
    gameInfo[0] = dealerPosition;
    gameInfo[1] = numberPlayersHand;
    gameInfo[2] = minGamesPlayed;

    return 0;
}

double median(double arr[], int arraySize)
{   //calculate the median of a double array and return it
    std::sort(&arr[0], &arr[arraySize]);
    double median = arraySize % 2 ? arr[arraySize / 2] : (arr[arraySize / 2 - 1] + arr[arraySize / 2]) / 2;
    return median;
}

int calcFamilyStats(double geneStats[familyCount * familyMembers][3], double zScores[familyCount * familyMembers], int generation)
{   //calcFamilyStats finds the mean and variance of each family's profits
    fstream geneStatsFile;
    geneStatsFile.open ("geneStatsFile.txt", fstream::in | fstream::out | fstream::app);
    geneStatsFile << generation << endl;
    for(int i = 0; i < familyCount * familyMembers; i ++)
    {
        geneStatsFile << geneStats[i][0] << "\t" << geneStats[i][2] << "\t" << zScores[i] << endl;
    }
    ///end of temp added files

    double medianVariance;

    fstream familyStatsFile;
    familyStatsFile.open ("familyStatsFile.txt", fstream::in | fstream::out | fstream::app);
    familyStatsFile << generation << endl;

    //calculate the median variance for each family and store it in a file "familyStatsFile"
    for(int i = 0; i < familyCount; i ++)
    {
        double familyVariances[familyMembers];
        for(int j = 0; j < familyMembers; j ++)
        {
            familyVariances[j] = geneStats[(i * familyMembers) + j][2];
        }
        medianVariance = median(familyVariances, familyMembers);

        familyStatsFile << medianVariance << "\t";
        cout << "Family " << i << " median variance is " << medianVariance << endl;
    }

    //calculate the median z score for each family and store it in a file "familyStatsFile"
    double medianZScore;
    for(int i = 0; i < familyCount; i ++)
    {
        double familyZScores[familyMembers];
        for(int j = 0; j < familyMembers; j ++)
        {
            familyZScores[j] = zScores[(i * familyMembers) + j];
        }
        medianZScore = median(familyZScores, familyMembers);

        familyStatsFile << medianZScore << "\t";
        cout << "Family " << i << " median Z score is " << medianZScore << endl;
    }

    familyStatsFile << endl;

    return 0;
}

int maxInIntArray(int arr[], int arraySize)
{   //maxInIntArray finds the largest integer in an array
    int maxElement = arr[0];
    for(int i = 1; i < arraySize; i ++)
    {
        if(arr[i] > maxElement)
        {
            maxElement = arr[i];
        }
    }
    return maxElement;
}

int geneFitnessMinTrials(double geneStats[familyCount * familyMembers][3], int gamesPlayed[familyCount * familyMembers], int generation)
{   //geneFitnessMinTrials calculates how many trials need to be calculated to get a good estimate of each family's fitness rankings
    double rankingZScore = NormalCDFInverse(0.5 + (geneFitnessRankingConfidence / 2));
    double upperFamilyMeanVariance, lowerFamilyMeanVariance;
    int trialsRequired[familyCount];
    double zScores[familyMembers];

    for(int i = 0; i < (familyCount * familyMembers); i ++)
    {
        //calculate the variance of each family member's profit
        geneStats[i][2] = (geneStats[i][1] / gamesPlayed[i]) - pow((geneStats[i][0] / gamesPlayed[i]), 2);
        //calculate the z score of each member
        zScores[i] = (geneStats[i][0] / gamesPlayed[i]) / pow(geneStats[i][2] / gamesPlayed[i], 0.5);
    }

    for(int i = 0; i < familyCount; i ++)
    {
        //sort family members
        int memberRanks[familyMembers];
        for(int j = 0; j < familyMembers; j ++)
        { //initially family members are given a rank but are not in order
            memberRanks[j] = j;
        }
        sortIntArrayByDouble(memberRanks, zScores, familyMembers, 0);

        int upperPercentileRank = floor((0.5 - geneFitnessRankingAccuracy) * (familyMembers - 1));
        int upperMemberRef = (i * familyMembers) + memberRanks[upperPercentileRank];
        int lowerPercentileRank = ceil((0.5 + geneFitnessRankingAccuracy) * (familyMembers - 1));
        int lowerMemberRef =  (i * familyMembers) + memberRanks[lowerPercentileRank];

        //find stats for difference between upper and lower ranked members
        double upperMean = geneStats[upperMemberRef][0] / gamesPlayed[upperMemberRef];
        double lowerMean = geneStats[lowerMemberRef][0] / gamesPlayed[lowerMemberRef];
        double upperStDev = sqrt(geneStats[upperMemberRef][2]);
        double lowerStDev = sqrt(geneStats[lowerMemberRef][2]);

        //calculate trials required for this family
        trialsRequired[i] = pow(rankingZScore * sqrt(2) / ((upperMean / upperStDev) - (lowerMean / lowerStDev)), 2.0);
    }

    //find maximum number of trials required
    int maxTrialsRequired = maxInIntArray(trialsRequired, familyCount);

    fstream fitnessTrialsFile;
    fitnessTrialsFile.open ("fitnessMinTrials.txt", fstream::in | fstream::out | fstream::app);
    fitnessTrialsFile << generation << endl;
    for(int i = 0; i < familyCount; i ++)
    {
        fitnessTrialsFile << trialsRequired[i] << "\t";
    }
    fitnessTrialsFile << endl;

    return maxTrialsRequired;
}

int saveLatestGeneStats(double geneStats[familyCount * familyMembers][3], double zScores[familyCount * familyMembers])
{   //save the variance and z score of the latest set of genes. Overwrite previous stats

    ofstream GeneFile("LatestGeneStats.txt");

    for(int i = 0; i < familyCount; i ++)
    {
        for(int j = 0; j < familyMembers; j ++)
        {
            int playerRef = (i * familyMembers) + j;
            GeneFile << playerRef << "\t" << zScores[playerRef] << "\t" << geneStats[playerRef][2] << endl;
        }
    }

    return 0;
}

double testGeneFitness(int minTrials, float bigBlind, float minChips, float maxChips, int layerSizes[numberLayers], double zScores[familyCount * familyMembers], int generation)
{   //testGeneFitness puts decision makers in hands with different numbers of players and chips to estimate its average profit. the Z score of the profit is used as the gene's fitness
    int chips[maxPlayers];
    int aiPlayers[maxPlayers];
    string playerNames[maxPlayers];
    int playersKnockedOut[maxPlayers];
    int numberPlayersHand, dealerPosition;
    double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize];
    int gamesPlayed[familyCount * familyMembers] = {0};
    double geneStats[familyCount * familyMembers][3]; //geneStats contains the sum of profit/loss, sum of square of profit/loss, and variance of profit
    int chipsBefore[maxPlayers];
    int playerRefNumbers[maxPlayers]; //the reference number of each player who play a given game
    int minGamesPlayed = 0; //initially no players have played, minGamesPlayed increases in the while loop
    int minTrialsNotReached = 1;


    //before each generation set players' geneStats to zero
    setToZero(geneStats);
    while(minTrialsNotReached)
    {
        while(minGamesPlayed < minTrials)
        {
            int gameInfo[3];
            setUpGame(bigBlind, maxChips, minChips, layerSizes, playerRefNumbers, gamesPlayed, chips, aiPlayers, playerNames, playersKnockedOut, playerWeights, gameInfo);
            dealerPosition = gameInfo[0];
            numberPlayersHand = gameInfo[1];
            minGamesPlayed = gameInfo[2];

            for(int i = 0; i < numberPlayersHand; i ++)
            {
                //record players' chips before the game
                chipsBefore[i] = chips[i];
            }

            playHand(dealerPosition, 1, aiPlayers, chips, playerNames, playersKnockedOut, numberPlayersHand, bigBlind, 0, playerWeights);
            //calculate the profit of each player
            int playerProfit[maxPlayers];

            for(int i = 0; i < numberPlayersHand; i ++)
            {
                //calculate profit
                playerProfit[i] = chips[i] - chipsBefore[i];
            }

            //update game stats for each gene
            for(int i = 0; i < numberPlayersHand; i ++)
            {
                geneStats[playerRefNumbers[i]][0] += (playerProfit[i] / bigBlind); //profit normalised with respect to big blind is used
                geneStats[playerRefNumbers[i]][1] += pow((playerProfit[i] / bigBlind), 2);
            }
        }

        ///minTrials = geneFitnessMinTrials(geneStats, gamesPlayed, generation);

        if(minTrials <= minGamesPlayed)
        {
            minTrialsNotReached = 0;

            int temp = geneFitnessMinTrials(geneStats, gamesPlayed, generation);
            fstream minTrialsFile;
            minTrialsFile.open ("minTrialsRecords.txt", fstream::in | fstream::out | fstream::app);
            minTrialsFile << temp <<endl;
        }
    }

    //calculate Z scores for each player
    for(int i = 0; i < (familyCount * familyMembers); i ++)
    {
        //calculate the variance of each gene's profit
        geneStats[i][2] = (geneStats[i][1] / gamesPlayed[i]) - pow((geneStats[i][0] / gamesPlayed[i]), 2);
        //calculate the z score of each member
        zScores[i] = (geneStats[i][0] / gamesPlayed[i]) / pow(geneStats[i][2] / gamesPlayed[i], 0.5);
    }

    //calculate and print the geometric mean variance for each family
    calcFamilyStats(geneStats, zScores, generation);

    saveLatestGeneStats(geneStats, zScores);

    /*for(int i = 0; i < (familyCount * familyMembers); i++)
    {
        cout << endl;
        cout << "Player ref " << i << endl;
        cout << "zScores is " << zScores[i] << endl;
        cout << "variance is " << geneStats[i][2] << endl;
        cout << "sample size is " << gamesPlayed[i] << endl;
    }*/
    return 0;
}

int sortFamilyRanks(int memberRanks[familyMembers], double allGeneFitness[familyCount * familyMembers], int familyNumber)
{   //sortFamilyRanks puts a given family's members in order of their fitness (greatest to least)
    //familyNumber of first family is 0

    double familyGeneFitness[familyMembers];

    //fill allGeneFitness with relevant figures
    for(int i = 0; i < familyMembers; i ++)
    {
        familyGeneFitness[i] = allGeneFitness[(familyMembers * familyNumber) + i];
    }

    int *rankedMembers = sortIntArrayByDouble(memberRanks, familyGeneFitness, familyMembers, 0);

    return 0;
}

int selectParents(int sortedMemberRanks[familyMembers], int newParents[2])
{   //selectParents chooses which members of a family will be used as an offspring's parent
    //the mother and father of an offspring must be different. The same parent can be used for multiple offspring
    //Only the better half of the family may be chosen as parents, each has an equal probability of being the parent
    int motherNumber, fatherNumber;
    int maxParentNumber = ((familyMembers - 1) / 2);
    if(maxParentNumber == 0)
    {
        cout << "Warning: there are not enough family members to select distinct parents" << endl;
    }

    //create the distribution for selecting parents
    std::uniform_int_distribution<int> randomParent(0, maxParentNumber);

    //select mother's number
    motherNumber = sortedMemberRanks[randomParent(generator)];

    //while father's number is not unique generate a father's number
    int fatherUnique = 0;
    while(!fatherUnique)
    {
        fatherNumber = sortedMemberRanks[randomParent(generator)];
        if(fatherNumber != motherNumber)
        {
            fatherUnique = 1;
        }
    }

    newParents[0] = motherNumber;
    newParents[1] = fatherNumber;

    return 0;
}

int createOffspring(int layerSizes[numberLayers], int familyNumber, int memberNumber, int rankedMembers[familyMembers], double crossoverRate, double mutationRate)
{   //createOffspring takes parents genes, mixes them and adds some mutation to create a child
    double offspringWeights[numberLayers][maxLayerSize][maxLayerSize]; //weights are stored in this array and stored in file at the end

    //define uniform distribution for new mutation genes
    std::normal_distribution<double> normalGeneDistribution(0.0, 1.0);

    //define distribution of genes mutating
    std::bernoulli_distribution mutationDistribution(mutationRate);

    //define distribution of gene crossover occurring
    std::bernoulli_distribution crossoverDistribution(crossoverRate);

    //select the parents of the offspring
    int parentNumbers[2];
    selectParents(rankedMembers, parentNumbers);

    int motherNumber = parentNumbers[0];
    int fatherNumber = parentNumbers[1];

    //open mother's file and copy genes to the offspring before adding father's genes and mutations
    stringstream ss1;
    ss1 << familyNumber;
    string motherFamilyString = ss1.str();
    stringstream ss2;
    ss2 << motherNumber;
    string motherMemberString = ss2.str();
    string motherFileName = "family";
    motherFileName.append(motherFamilyString);
    motherFileName.append("member");
    motherFileName.append(motherMemberString);
    motherFileName.append(".txt");
    ifstream motherWeightsFile( motherFileName.c_str() );

    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        for(int j = 0; j < layerSizes[i]; j ++)
        {
            for(int k = 0; k < layerSizes[i + 1]; k ++)
            {
                motherWeightsFile >> offspringWeights[i][j][k];
            }
        }
    }
    motherWeightsFile.close();

    //open father's file
    stringstream ss3;
    ss3 << familyNumber;
    string familyString = ss3.str();
    stringstream ss4;
    ss4 << fatherNumber;
    string memberString = ss4.str();
    string fatherFileName = "family";
    fatherFileName.append(familyString);
    fatherFileName.append("member");
    fatherFileName.append(memberString);
    fatherFileName.append(".txt");
    ifstream fatherWeightsFile( fatherFileName.c_str() );

    //go through father's gene, if crossover is done then use father's gene, otherwise assign father's gene to a temp
    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        double doubleLayerSize = layerSizes[i];
        for(int j = 0; j < layerSizes[i]; j ++)
        {
            for(int k = 0; k < layerSizes[i + 1]; k ++)
            {
                double temp;
                if(mutationDistribution(generator))
                {
                    fatherWeightsFile >> temp;
                    offspringWeights[i][j][k] = (normalGeneDistribution(generator))/ sqrt(doubleLayerSize); //divide by sqrt(layersize) to avoid saturating preceptrons
                }
                else
                {
                    if(crossoverDistribution(generator))
                    {
                        fatherWeightsFile >> offspringWeights[i][j][k];
                    }
                    else
                    {
                        //if not doing crossover nothing needs to be changed in the offspring's genes
                        fatherWeightsFile >> temp;
                    }
                }
            }
        }
    }

    //open offspring's file
    stringstream ss5;
    ss5 << familyNumber;
    string offspringFamilyString = ss5.str();
    stringstream ss6;
    ss6 << memberNumber;
    string offspringMemberString = ss6.str();
    string offspringFileName = "family";
    offspringFileName.append(offspringFamilyString);
    offspringFileName.append("member");
    offspringFileName.append(offspringMemberString);
    offspringFileName.append(".txt");
    ofstream offspringWeightsFile(offspringFileName.c_str() );

    //store new offspring weights in file
    for(int i = 0; i < (numberLayers - 1); i ++)
    {
        for(int j = 0; j < layerSizes[i]; j ++)
        {
            for(int k = 0; k < layerSizes[i + 1]; k ++)
            {
                offspringWeightsFile << offspringWeights[i][j][k] << "\t";
            }
            offspringWeightsFile << "\n";
        }
    }

    return 0;
}

int updateFamily(double allGeneFitness[familyCount * familyMembers], int familyNumber, double crossoverRate, double mutationRate, int layerSizes[numberLayers])
{   //updateFamily replaces the worse half of a family with new offspring
    int memberRanks[familyMembers];
    for(int i = 0; i < familyMembers; i ++)
    { //initially family members are given a rank but are not in order
        memberRanks[i] = i;
    }
    sortFamilyRanks(memberRanks, allGeneFitness, familyNumber);

    int maxParentNumber = ((familyMembers - 1) / 2);

    //replace all family members after the maximum parent position
    for(int i = (maxParentNumber + 1); i < familyMembers; i ++)
    {
        //select offspring's parents
        int newParents[2];
        selectParents(memberRanks, newParents);

        //change the offspring's file
        ///createOffspring(layerSizes, familyNumber, memberRanks[i], memberRanks, crossoverRate, mutationRate);
    }

    return 0;
}

int updateGenes(double allGeneFitness[familyCount * familyMembers], double crossoverRate, double mutationRates[familyCount], int layerSizes[numberLayers])
{   //updateGenes performs the genetic algorithm update for genes from all families
    for(int i = 0; i < familyCount; i ++)
    {
        updateFamily(allGeneFitness, i, crossoverRate, mutationRates[i], layerSizes);
    }
    return 0;
}

int doGeneticAlgorithm(int numberGenerations, int epochLength, int minTrials, double crossoverRate, double minMutationRate, double maxMutationRate, int bigBlind, int minChips, int maxChips, int layerSizes[numberLayers])
{   //doGeneticAlgorithms repeatedly tests gene fitness and updates genes
    //print time
    time_t  timev;
    time(&timev);
    cout << "Genetic algorithm start time " << timev << endl;

    double zScores[familyCount * familyMembers];

    //set the mutation rates of each family
    double mutationRates[familyCount]; //the mutation rates of each family
    for(int i = 0; i < familyCount; i ++)
    {
        mutationRates[i] = minMutationRate + (i * ((maxMutationRate - minMutationRate) / (familyCount - 1)));
    }

    for(int generationCount = 1; generationCount <= numberGenerations; generationCount ++)
    {
        cout << endl << "Generation number " << generationCount << endl;
        //test the Z score of long term profit for each genome
        testGeneFitness(minTrials, bigBlind, minChips, maxChips, layerSizes, zScores, generationCount);

        //update each family's genes
        updateGenes(zScores, crossoverRate, mutationRates, layerSizes);

        if((generationCount % epochLength) == 0)
        {
            //do stuff at the end of epoch
            ///placeholder for future use
        }
    }

    return 0;
}

int playAgainstAI(int playerRefNumbers[maxPlayers], string humanName, int manualDealing, int maxNumberHands, int layerSizes[numberLayers])
{   //playAgainstAI puts one human player against selected AI players
    //human player must be in position 0 of playerRefNumbers[]
    //the first position in playerRefNumbers[] which has no player should be given a value of -1. E.g. playerRefNumbers[maxplayers] = {0,2,15,7,-1,0,0,0}; puts refs 2,15,7 against human

    //count the number of players
    int numberPlayersHand = 1; //initially one human player, add on players until a reference number of -1 is found
    int initialChips = 10000;
    int bigBlind = 100;
    int trainingMode = 0;
    int initialPosition = 0;

    int aiPlayers[maxPlayers];
    int playersKnockedOut[maxPlayers];
    int chips[maxPlayers];

    for(int i = 1; i < maxPlayers; i ++)
    {
        if(playerRefNumbers[i] == -1)
        {
            i = maxPlayers;
        }
        else
        {
            numberPlayersHand ++;
        }
    }

    //fill the weights array with each AI players' weights
    double playerWeights[maxPlayers][numberLayers][maxLayerSize][maxLayerSize];

    for(int playerCount = 1; playerCount < numberPlayersHand; playerCount ++)
    {//populate the weights array for each player
        populateWeightsArray(playerCount, playerRefNumbers[playerCount], layerSizes, playerWeights);
    }

    //create players' names for display during the game
    string playerNames[maxPlayers];
    playerNames[0] = humanName;
    for(int i = 1; i < maxPlayers; i ++)
    {
        stringstream ss3;
        ss3 << playerRefNumbers[i];
        string refString = ss3.str();
        string nameString = "Ref";
        nameString.append(refString);
        playerNames[i] = nameString;
    }

    //set playersKnockedOut, aiPlayers, chips for each player
    for(int i = 0; i < maxPlayers; i ++)
    {
        aiPlayers[i] = 1;
        if(i < numberPlayersHand)
        {
            playersKnockedOut[i] = 0;
            chips[i] = initialChips;
        }
        else
        {
            playersKnockedOut[i] = 1;
            chips[i] = 0;
        }
    }
    aiPlayers[0] = 0; //set first player as human

    playManyHands(bigBlind, manualDealing, trainingMode, maxNumberHands, initialPosition, playerNames, aiPlayers, chips, playersKnockedOut, playerWeights);

    return 0;
}

int main()
{
    int learnFromScratch = 0; //if learnFromScratch is 1 the files containing gene weights are assumed to be empty. If 0 then exiting genetic information in files is used
    int minNumberTrials = 8000; //the minimum number of hands each gene must play to estimate their performance
    double crossoverRate = 0.5, minMutationRate = 0.05, maxMutationRate = 0.3;
    int numberGenerations = 1, epochLength = 1;
    float minChips = 10, maxChips = 200; //the range of chips (relative to big blind) which players can have in a game
    int bigBlind = 100;
    int layerSizes[numberLayers] = {inputLayerSize, hiddenLayerSize, outputLayerSize, 1};
    ///int oldLayerSizes[numberLayers] = {12, 9, 3, 1};

    ///addNewWeights(layerSizes, oldLayerSizes, numberLayers);

    //if the algorithm is learning from scratch create the files storing player information
    if(learnFromScratch == 1)
    {
        cout << "Warning! Previous genetic weights are to be deleted" << endl;
        cout << "Enter anything to overwrite neural network weights and continue" << endl;
        string temp;
        cin >> temp;
        createGeneFiles(layerSizes);
    }

    ///line 3379 updating genes is commented out for testing
    doGeneticAlgorithm(numberGenerations, epochLength, minNumberTrials, crossoverRate, minMutationRate, maxMutationRate, bigBlind, minChips, maxChips, layerSizes);

    ///int playerRefNumbers[maxPlayers] = {0,3,12,42,-1,0,0,0};
    ///playAgainstAI(playerRefNumbers, "Hugh", 1, 20, layerSizes);

    return 0;
}

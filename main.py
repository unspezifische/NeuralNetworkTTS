##******************************************************************************
## This is a Python3 port of the pink-trombone porject originally release by MIT
## Some additional bells and whistles have been added, such as:
##      - tensorflow control of sound generation functions
##      - A training function for the tensorflow network
##      - A playback function what takes a string as argument outputs via pyAudio
##
## Originally devloped by
## Ported to Python by Ian Johnson
##
## This program is composed of 3 different sections-
##  1. The core sound generating functions that mathematically replicate human speech (based on tensorflow input)
##  2. A tensorflow training module that takes
##
## TODO:
##  1. determine input values for pink-trombone module (obfuscate to seperate module, or include for fewer dependencies?)
##  2. setup training function for tensorflow that accepts 
##
## Algorithm:
##  1.

from math import *  # hell yeah we need self!
import random       # provides access to random.random()
import noise        # provides simplex noise functions
import pyAudio      # allows playback of sounds created

import tensorflow as tf     ## import the tensorflow modules

## Used to get the lowest of 3 given values
def clamp(number, minVal, maxVal):
    if (number < minVal):
        return minVal
    elif (number > maxVal):
        return maxVal
    else:
        return number

## Used to determine the direction a value is "trending"?
def moveTowards(current, target, amount):
    if (current < target):
        return min(current + amount, target)
    else:
        return max(current-amount, target)

## Second implmentation of above function
def moveTowards(current, target, amountUp, amountDown):
    if (current < target):
        return min(current+amountUp, target)
    else:
        return max(current-amountDown, target)

## gaussian
def gaussian():
    s = 0
    c = 0   ## for loop itertor
    for c in range(16): ## loop 16 times (0-16)
        s += random.random()    ## add a float between 0.0 & 1.0 to s
        c += 1  ## increment iterator
    return (s-8)/4  ## idk why it does this. i'm just the scribe


# define callback
def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)   ## not this
    return (data, pyaudio.paContinue)


sampleRate = 0      ## 1 second of sound. this gets used in startSound()
time = 0            ## makes sense
temp = {a:0, b:0}   ## not sure what self dict is for
alwaysVoice = False ## probably unnecessary
autoWobble = False  ## probably unnecessary
noiseFreq = 500
noiseQ = 0.7


## TODO- rewrite this to use pyAudio (using "callback mode")
class AudioSystem:
    ## blockLength Can be set to 512 for more responsiveness but
    ## potential crackling if CPU can't fill the buffer fast enough (latency)
    blockLength = 2048      ## this will be the frame frameCount parameter of callback
    blockTime = 1
    started = False
    soundOn = False

    def __init__():
        sampleRate = self.audioContext.sampleRate       ## set sample rate to something useful
        self.blockTime = self.blockLength/sampleRate    ## set blockTime to something useful

    ## TODO rewrite this to use whatever library
    def startSound():
        this.scriptProcessor.onaudioprocess = AudioSystem.doScriptProcessor     # old line

        whiteNoise = this.createWhiteNoiseNode(2 * sampleRate) ## 2 seconds of noise

        aspirateFilter = this.audioContext.createBiquadFilter()
        aspirateFilter.type = "bandpass"
        aspirateFilter.frequency.value = 500
        aspirateFilter.Q.value = 0.5
        whiteNoise.connect(aspirateFilter)
        aspirateFilter.connect(this.scriptProcessor)

        fricativeFilter = this.audioContext.createBiquadFilter()
        fricativeFilter.type = "bandpass"
        fricativeFilter.frequency.value = 1000
        fricativeFilter.Q.value = 0.5
        whiteNoise.connect(fricativeFilter)
        fricativeFilter.connect(this.scriptProcessor)

        whiteNoise.start(0)

    ## TODO fix this to use python library (create a buffer of whatever length and fill it with white noise)
    def createWhiteNoiseNode(frameCount):
        myArrayBuffer = this.audioContext.createBuffer(1, frameCount, sampleRate)

        nowBuffering = myArrayBuffer.getChannelData(0)  ## Get the audio buffer object
        i = 0
        for i in range(frameCount):
            nowBuffering[i] = random()## gaussian()
            i += 1
        source = this.audioContext.createBufferSource()
        source.buffer = myArrayBuffer
        source.loop = true

        return source

    ## This function might not be necessary after rework
    def doScriptProcessor(event):
        ## Get input arrays from AudioBuffer
        inputArray1 = event.inputBuffer.getChannelData(0)
        vainputArray2 = event.inputBuffer.getChannelData(1)
        outArray = event.outputBuffer.getChannelData(0)
        j = 0
        N = len(outArray)  ## This variable gets used within the loop
        for j in range(N):
            lambda1 = j/N
            lambda2 = (j+0.5)/N
            glottalOutput = Glottis.runStep(lambda1, inputArray1[j])

            vocalOutput = 0
            ## Tract runs at twice the sample rate
            Tract.runStep(glottalOutput, inputArray2[j], lambda1)
            vocalOutput += Tract.lipOutput + Tract.noseOutput
            Tract.runStep(glottalOutput, inputArray2[j], lambda2)
            vocalOutput += Tract.lipOutput + Tract.noseOutput
            outArray[j] = vocalOutput * 0.125

        Glottis.finishBlock()
        Tract.finishBlock()

    def mute():
        ## TODO- make this the Sonic Pi mute command
        return

    def unmute():
        ## TODO- make this the Sonic Pi unmute command
        return


## Glottis needs to be a class object for some of the calls to work as written
class Glottis:
    '''
    this class defines the framework for speech synthesis
    '''
    timeInWaveform = 0
    oldFrequency = 140
    newFrequency = 140
    UIFrequency = 140
    smoothFrequency = 140
    oldTenseness = 0.6
    newTenseness = 0.6
    UITenseness = 0.6
    totalTime = 0
    vibratoAmount = 0.005
    vibratoFrequency = 6
    intensity = 0
    loudness = 1
    isTouched = False
    isTouchingSomewhere = False
    ctx = backCtx
    x = 240     ## Are these necessary if no GUI?
    y = 530

    semitones = 20
    marks = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    baseNote = 87.3071 ## F

    def __init__():
        setupWaveform(0)

    # TODO make sense of this
    def handleTouch():
        semitone = self.semitones * local_x / self.keyboardWidth + 0.5
        Glottis.UIFrequency = self.baseNote * pow(2, semitone/12)
        if (Glottis.intensity == 0):
             Glottis.smoothFrequency = Glottis.UIFrequency
        # Glottis.UIRd = 3*local_y / (self.keyboardHeight-20)
        t = clamp(1-local_y / (self.keyboardHeight-28), 0, 1)
        Glottis.UITenseness = 1-cos(t * pi * 0.5)
        Glottis.loudness = pow(Glottis.UITenseness, 0.25)

    def runStep(Lambda, noiseSource):
        timeStep = 1.0 / sampleRate
        self.timeInWaveform += timeStep
        self.totalTime += timeStep
        if (self.timeInWaveform > self.waveformLength):
          self.timeInWaveform -= self.waveformLength
          self.setupWaveform(Lambda)
        out = self.normalizedLFWaveform(self.timeInWaveform / self.waveformLength)
        aspiration = self.intensity*(1-sqrt(self.UITenseness)) * self.getNoiseModulator() * noiseSource
        aspiration *= 0.2 + 0.02 * noise.simplex1(self.totalTime * 1.99)
        out += aspiration
        return out

    def getNoiseModulator():
        voiced = 0.1 + 0.2 * max(0, sin(pi * 2 * self.timeInWaveform / self.waveformLength))
        # return 0.3
        return self.UITenseness * self.intensity * voiced + (1-self.UITenseness * self.intensity ) * 0.3

    def finishBlock():
        vibrato = 0
        vibrato += self.vibratoAmount * sin(2 * pi * self.totalTime *self.vibratoFrequency)
        vibrato += 0.02 * noise.simplex1(self.totalTime * 4.07)
        vibrato += 0.04 * noise.simplex1(self.totalTime * 2.15)
        if (autoWobble):
            vibrato += 0.2 * noise.simplex1(self.totalTime * 0.98)
            vibrato += 0.4 * noise.simplex1(self.totalTime * 0.5)

        if (self.UIFrequency>self.smoothFrequency):
            self.smoothFrequency = min(self.smoothFrequency * 1.1, self.UIFrequency)
        if (self.UIFrequency<self.smoothFrequency):
            self.smoothFrequency = max(self.smoothFrequency / 1.1, self.UIFrequency)
        self.oldFrequency = self.newFrequency
        self.newFrequency = self.smoothFrequency * (1+vibrato)
        self.oldTenseness = self.newTenseness
        self.newTenseness = self.UITenseness + 0.1*noise.simplex1(self.totalTime*0.46)+0.05*noise.simplex1(self.totalTime*0.36)
        if (not self.isTouched and (alwaysVoice or self.isTouchingSomewhere)):
            self.newTenseness += (3-self.UITenseness) * (1-self.intensity)

        if (self.isTouched or alwaysVoice or self.isTouchingSomewhere):
            self.intensity += 0.13
        else:
            self.intensity -= (AudioSystem.blockTime * 5)
        self.intensity = clamp(self.intensity, 0, 1)


    def setupWaveform(Lambda):
        self.frequency = self.oldFrequency * (1 - Lambda) + self.newFrequency * Lambda
        tenseness = self.oldTenseness * (1 - Lambda) + self.newTenseness * Lambda
        self.Rd = 3*(1 - tenseness)
        self.waveformLength = 1.0/self.frequency

        Rd = self.Rd
        if (Rd<0.5):
            Rd = 0.5
        if (Rd>2.7):
            Rd = 2.7
        # output  ## define the variable, for what?
        ## normalized to time = 1, Ee = 1
        Ra = -0.01 + 0.048 * Rd
        Rk = 0.224 + 0.118 * Rd
        Rg = (Rk/4) * (0.5 + 1.2 * Rk)/(0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

        Ta = Ra
        Tp = 1 / (2*Rg)
        Te = Tp + Tp*Rk ##

        epsilon = 1/Ta
        shift = exp(-epsilon * (1-Te))  ## TODO- is JS Math.exp same as python3's exp or pow?
        Delta = 1 - shift ##divide by self to scale RHS

        RHSIntegral = (1/epsilon) * (shift - 1) + (1-Te) * shift
        RHSIntegral = RHSIntegral/Delta

        totalLowerIntegral = - (Te-Tp)/2 + RHSIntegral
        totalUpperIntegral = -totalLowerIntegral

        omega = PI/Tp
        s = sin(omega*Te)
        ## need E0*e^(alpha*Te)*s = -1 (to meet the return at -1)
        ## and E0*e^(alpha*Tp/2) * Tp*2/pi = totalUpperIntegral
        ##             (our approximation of the integral up to Tp)
        ## writing x for e^alpha,
        ## have E0*x^Te*s = -1 and E0 * x^(Tp/2) * Tp*2/pi = totalUpperIntegral
        ## dividing the second by the first,
        ## letting y = x^(Tp/2 - Te),
        ## y * Tp*2 / (pi*s) = -totalUpperIntegral
        y = -PI*s*totalUpperIntegral / (Tp*2)
        z = log(y)
        alpha = z/(Tp/2 - Te)
        E0 = -1 / (s*exp(alpha*Te))
        self.alpha = alpha
        self.E0 = E0
        self.epsilon = epsilon
        self.shift = shift
        self.Delta = Delta
        self.Te=Te
        self.omega = omega

    def normalizedLFWaveform(t):
        if (t>self.Te):
            output = (-exp(-self.epsilon * (t-self.Te)) + self.shift)/self.Delta
        else:
            output = self.E0 * exp(self.alpha*t) * sin(self.omega * t)

        return output * self.intensity * self.loudness


class Tract:
    n = 44
    bladeStart = 10
    tipStart = 32
    lipStart = 39
    R = [] ## component going right
    L = [] ## component going left
    reflection = []
    junctionOutputR = []
    junctionOutputL = []
    maxAmplitude = []
    diameter = []
    restDiameter = []
    targetDiameter = []
    newDiameter = []
    A = []
    glottalReflection = 0.75
    lipReflection = -0.85
    lastObstruction = -1
    fade = 1.0 ## 0.9999
    movementSpeed = 15 ## cm per second
    transients = []
    lipOutput = 0
    noseOutput = 0
    velumTarget = 0.01

    def __init__():
        this.bladeStart = floor(this.bladeStart*this.n/44)
        this.tipStart = floor(this.tipStart*this.n/44)
        this.lipStart = floor(this.lipStart*this.n/44)
        this.diameter = [this.n]        ## Flaot64Array is array of 64-bit floating point numbers
        this.restDiameter = [this.n]    ## In python, arrays can hold any type
        this.targetDiameter = [this.n]  ## Also, floats in python are 64-bit by default
        this.newDiameter = [this.n]     ## QED- this is just an array of floats. nothing special

        i = 0
        for i in range(this.n):
            diameter = 0
            if (i<7*this.n/44-0.5):
                diameter = 0.6
            elif (i<12*this.n/44):
                diameter = 1.1
            else:
                diameter = 1.5
            this.diameter[i] = this.restDiameter[i] = this.targetDiameter[i] = this.newDiameter[i] = diameter
            i += 1

        this.R = [this.n]
        this.L = [this.n]
        this.reflection = [this.n+1]
        this.newReflection = [this.n+1]
        this.junctionOutputR = [this.n+1]
        this.junctionOutputL = [this.n+1]
        this.A = [this.n]
        this.maxAmplitude = [this.n]

        this.noseLength = floor(28*this.n/44)
        this.noseStart = this.n-this.noseLength + 1
        this.noseR = [this.noseLength]
        this.noseL = [this.noseLength]
        this.noseJunctionOutputR = [this.noseLength + 1]
        this.noseJunctionOutputL = [this.noseLength + 1]
        this.noseReflection = [this.noseLength + 1]
        this.noseDiameter = [this.noseLength]
        this.noseA = [this.noseLength]
        this.noseMaxAmplitude = [this.noseLength]

        i = 0
        for i in range(this.noseLength):
            diameter
            d = 2*(i/this.noseLength)
            if (d<1):
                diameter = 0.4 + 1.6 * d
            else:
                diameter = 0.5 + 1.5 * (2 - d)
            diameter = min(diameter, 1.9)
            this.noseDiameter[i] = diameter
            i += 1

        this.newReflectionLeft = this.newReflectionRight = this.newReflectionNose = 0 ## does this even work in Python?
        this.calculateReflections()
        this.calculateNoseReflections()
        this.noseDiameter[0] = this.velumTarget


    def reshapeTract(deltaTime):
        amount = deltaTime * this.movementSpeed
        newLastObstruction = -1

        i = 0
        for i in range(this.n):
            diameter = this.diameter[i]
            targetDiameter = this.targetDiameter[i]
            if (diameter <= 0):
                newLastObstruction = i
            # slowReturn  ## define variable, unnecessary in Python
            if (i<this.noseStart):
                slowReturn = 0.6
            elif (i >= this.tipStart):
                slowReturn = 1.0
            else:
                slowReturn = 0.6 + 0.4 * (i - this.noseStart)/(this.tipStart - this.noseStart)
            this.diameter[i] = moveTowards(diameter, targetDiameter, slowReturn*amount, 2*amount)
        if (this.lastObstruction > -1 and newLastObstruction == -1 and this.noseA[0] < 0.05):
            this.addTransient(this.lastObstruction)
        this.lastObstruction = newLastObstruction
        i += 1

        amount = deltaTime * this.movementSpeed
        this.noseDiameter[0] = moveTowards(this.noseDiameter[0], this.velumTarget, amount * 0.25, amount * 0.1)
        this.noseA[0] = this.noseDiameter[0] * this.noseDiameter[0]

    def calculateReflections():
        i = 0
        for i in range(this.n):
            this.A[i] = this.diameter[i]*this.diameter[i] ##ignoring PI etc.
            i += 1

        i = 1
        for i in range(this.n):
            this.reflection[i] = this.newReflection[i]
            if (this.A[i] == 0):
                this.newReflection[i] = 0.999 ##to prevent some bad behaviour if 0
            else:
                this.newReflection[i] = (this.A[i-1]-this.A[i]) / (this.A[i-1]+this.A[i])
            i += 1

        ## now at junction with nose

        this.reflectionLeft = this.newReflectionLeft
        this.reflectionRight = this.newReflectionRight
        this.reflectionNose = this.newReflectionNose
        sumTotal = this.A[this.noseStart] + this.A[this.noseStart + 1] + this.noseA[0]
        this.newReflectionLeft = (2 * this.A[this.noseStart] - sumTotal)/sumTotal
        this.newReflectionRight = (2 * this.A[this.noseStart + 1] - sumTotal)/sumTotal
        this.newReflectionNose = (2 * this.noseA[0] - sumTotal)/sumTotal


    def calculateNoseReflections():
        i = 0
        for i in range(this.noseLength):
            this.noseA[i] = this.noseDiameter[i]*this.noseDiameter[i]
            i += 1

        i = 1
        for i in range(this.noseLength):
            this.noseReflection[i] = (this.noseA[i-1]-this.noseA[i]) / (this.noseA[i-1]+this.noseA[i])
            i += 1


    ## Had to change the 'lambda' parameter to Lambda because 'lambda' is a reserved word in Python
    def runStep(glottalOutput, turbulenceNoise, Lambda):
        updateAmplitudes = (random.random() < 0.1)

        ## mouth
        this.processTransients()
        this.addTurbulenceNoise(turbulenceNoise)

        ## this.glottalReflection = -0.8 + 1.6 * Glottis.newTenseness
        this.junctionOutputR[0] = this.L[0] * this.glottalReflection + glottalOutput
        this.junctionOutputL[this.n] = this.R[this.n-1] * this.lipReflection

        i = 1
        for i in range(this.n):
          r = this.reflection[i] * (1-Lambda) + this.newReflection[i]*Lambda
          w = r * (this.R[i-1] + this.L[i])
          this.junctionOutputR[i] = this.R[i-1] - w
          this.junctionOutputL[i] = this.L[i] + w
          i += 1

        ## now at junction with nose
        i = this.noseStart
        r = this.newReflectionLeft * (1-Lambda) + this.reflectionLeft*Lambda
        this.junctionOutputL[i] = r * this.R[i-1] + (1+r) * (this.noseL[0] + this.L[i])
        r = this.newReflectionRight * (1-Lambda) + this.reflectionRight * Lambda
        this.junctionOutputR[i] = r * this.L[i] + (1+r) * (this.R[i-1] + this.noseL[0])
        r = this.newReflectionNose * (1-Lambda) + this.reflectionNose * Lambda
        this.noseJunctionOutputR[0] = r * this.noseL[0] + (1+r) * (this.L[i] + this.R[i-1])

        i = 0
        for i in range (this.n):
            ## this.R[i] = this.junctionOutputR[i] * 0.999
            ## this.L[i] = this.junctionOutputL[i+1] * 0.999

            this.R[i] = clamp(this.junctionOutputR[i] * 0.999, -1, 1)
            this.L[i] = clamp(this.junctionOutputL[i+1] * 0.999, -1, 1)

            if (updateAmplitudes):
                amplitude = abs(this.R[i]+this.L[i])
                if (amplitude > this.maxAmplitude[i]):
                    this.maxAmplitude[i] = amplitude
                else:
                    this.maxAmplitude[i] *= 0.999
            i += 1

        this.lipOutput = this.R[this.n-1]

        ## nose
        this.noseJunctionOutputL[this.noseLength] = this.noseR[this.noseLength-1] * this.lipReflection

        i = 1
        for i in range(this.noseLength):
            w = this.noseReflection[i] * (this.noseR[i-1] + this.noseL[i])
            this.noseJunctionOutputR[i] = this.noseR[i-1] - w
            this.noseJunctionOutputL[i] = this.noseL[i] + w
            i += 1

        i = 0
        for i in range(this.noseLength):
          ## this.noseR[i] = this.noseJunctionOutputR[i] * this.fade
          ## this.noseL[i] = this.noseJunctionOutputL[i+1] * this.fade

          this.noseR[i] = clamp(this.noseJunctionOutputR[i] * 0.999, -1, 1)
          this.noseL[i] = clamp(this.noseJunctionOutputL[i+1] * 0.999, -1, 1)

          if (updateAmplitudes):
            amplitude = abs(this.noseR[i]+this.noseL[i])
            if (amplitude > this.noseMaxAmplitude[i]):
                this.noseMaxAmplitude[i] = amplitude
            else:
                this.noseMaxAmplitude[i] *= 0.999
          i += 1

        this.noseOutput = this.noseR[this.noseLength-1]

    def finishBlock():
        this.reshapeTract(AudioSystem.blockTime)
        this.calculateReflections()

    def addTransient(position):
        trans = {}
        trans.position = position
        trans.timeAlive = 0
        trans.lifeTime = 0.2
        trans.strength = 0.3
        trans.exponent = 200
        this.transients.push(trans)

    def processTransients():
        i = 0
        for i in range(this.transients.length):
            trans = this.transients[i]
            amplitude = trans.strength * pow(2, -trans.exponent * trans.timeAlive)
            this.R[trans.position] += amplitude/2
            this.L[trans.position] += amplitude/2
            trans.timeAlive += 1.0/(sampleRate * 2)

            i += 1

        i = this.transients.length - 1
        while (i>=0):       ## This serves better as a while loop since number of iterations is not specific
            trans = this.transients[i]
            if (trans.timeAlive > trans.lifeTime):
                this.transients.splice(i,1)
            i -= 1

    def addTurbulenceNoise(turbulenceNoise):
        j = 0
        for j in range(UI.touchesWithMouse.length):     ##TODO-  Didn't UI class go away? What replaces it as input?
            touch = UI.touchesWithMouse[j]
            if (touch.index<2 or touch.index>Tract.n):
                continue
            if (touch.diameter<=0):
                continue
            intensity = touch.fricative_intensity
            if (intensity == 0):
                continue
            this.addTurbulenceNoiseAtIndex(0.66*turbulenceNoise*intensity, touch.index, touch.diameter)
        j += 1

    def addTurbulenceNoiseAtIndex(turbulenceNoise, index, diameter):
        i = floor(index)
        delta = index - i
        turbulenceNoise *= Glottis.getNoiseModulator()
        thinness0 = clamp(8*(0.7-diameter),0,1)
        openness = clamp(30*(diameter-0.3), 0, 1)
        noise0 = turbulenceNoise*(1-delta)*thinness0*openness
        noise1 = turbulenceNoise*delta*thinness0*openness
        this.R[i+1] += noise0/2
        this.L[i+1] += noise0/2
        this.R[i+2] += noise1/2
        this.L[i+2] += noise1/2

def start():
    ## initialize class objects
    AudioSystem()
    Glottis()
    Tract()

    def redraw():
        time = Date.now/1000    ## TODO update for Python?



## On click, activate audio

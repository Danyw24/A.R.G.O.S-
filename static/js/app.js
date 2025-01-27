
// Audio setup

const beepSound = new Audio('static/assets/beep.mp3');
let beepInterval;


// Global variables
let peerConnection = null;
let dataChannel = null;
let timerInterval = null;
let seconds = 0;
let isConnected = false;


// DOM elements
const ringBox = document.getElementById('ringBox');
const callButton = document.getElementById('callButton');
const endCallBtn = document.getElementById('endCallBtn');
const callStatus = document.querySelector('.call-status');


const iceServers = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
    ]
}

async function startCall() {
    ringBox.style.display = 'block';
    callStatus.textContent = 'Iniciando';
    startBeeping();
    await initOpenAIRealtime();
}


function startBeeping() {
    beepSound.play();
    beepInterval = setInterval(() => {
        beepSound.play();
    }, 3000);
}


function stopBeeping() {
    clearInterval(beepInterval);
}

// fix an ispect this
const fns = {
    ask: async ({  message }) => {
        try {
            ask(message);
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
};


async function initOpenAIRealtime() {
    try {
        const tokenResponse = await fetch("session");
        const data = await tokenResponse.json();
        const EPHEMERAL_KEY = data.client_secret.value;

        peerConnection = new RTCPeerConnection(iceServers);
        
        // Add connection state change listener
        peerConnection.onconnectionstatechange = (event) => {
            console.log("Connection state:", peerConnection.connectionState);
            if (peerConnection.connectionState === 'connected') {
                stopBeeping();
                isConnected = true; 
                callStatus.textContent = 'Connected';
                endCallBtn.style.display = 'block';
            }
        };

        // Setup audio
        const audioElement = document.createElement("audio");
        audioElement.autoplay = true;
        peerConnection.ontrack = event => {
            audioElement.srcObject = event.streams[0];
        };

        const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        peerConnection.addTrack(mediaStream.getTracks()[0]);

        // Create data channel after peerConnection is initialized
        dataChannel = peerConnection.createDataChannel('response');

        // Configure data channel functions
        function configureData() {
            console.log('Configuring data channel');
            const event = {
                type: 'session.update',
                session: {
                    modalities: ['text', 'audio'],
                    tools: [
                        {
                            type: 'function',
                            name: 'ask',
                            description: 'ask to agency and then return the response to user',
                            parameters: {
                                type: 'object',
                                properties: {
                                    
                                    message: { 
                                        type: 'string', 
                                        description: 'message content to ask to agency and then return the response to user' 
                                    }
                                },
                                required: ['message']
                            }
                        }
                    ]
                }
            };
            dataChannel.send(JSON.stringify(event));
        }

        // Add event listeners after data channel is created
        dataChannel.addEventListener('open', () => {
            console.log('Data channel opened');
            configureData();
        
        });

        dataChannel.addEventListener('message', async (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                console.log(msg);
                if (msg.type === 'response.function_call_arguments.done') {
                    const fn = fns[msg.name];
                    if (fn !== undefined) {
                        console.log(`Calling function ${msg.name} with arguments:`, msg.arguments);
                        const args = JSON.parse(msg.arguments);
                        console.log(args.message);
                        const result = await ask(args.message);
                        
                        const event = {
                            type: 'conversation.item.create',
                            item: {
                                type: 'function_call_output',
                                call_id: msg.call_id,
                                output: result
                            }
                        };
                        console.log("Result:" , result);
                        dataChannel.send(JSON.stringify(event));
                    }
                }
            } catch (error) {
                console.error('Error handling message:', error);
            }
        });

        // Create and send offer
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        const apiUrl = "https://api.openai.com/v1/realtime";
        const model = "gpt-4o-realtime-preview-2024-12-17";
        
        const sdpResponse = await fetch(`${apiUrl}?model=${model}`, {
            method: "POST",
            body: offer.sdp,
            headers: {
                Authorization: `Bearer ${EPHEMERAL_KEY}`,
                "Content-Type": "application/sdp"
            },
        });

        const answer = {
            type: "answer",
            sdp: await sdpResponse.text(),
        };
        await peerConnection.setRemoteDescription(answer);

    } catch (error) {
        console.error("Error:", error);
        endCall();
    }
}


function formatCallSummary(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    
    let summary = 'Call Duration: ';
    if (hours > 0) summary += `${hours}h `;
    if (minutes > 0) summary += `${minutes}m `;
    summary += `${remainingSeconds}s`;
    
    return summary;
}


async function ask(message){
    try {
        const loader = document.querySelector('.loader');
        loader.style.display = 'block';
    
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if(!response.ok){
            console.error("Error en solicitud /ask");
        }

        const responseData = await response.json();
        const AgencyResponce = responseData.response + ", A continuacion informa al usuario de la respuesta de la agencia"; 
        loader.style.display = 'none';
        // insertar audio de respuesta -->
        
        

        return AgencyResponce;


    } catch (error) {
        console.error('Error sending Question: ', error);
        return "Error"
    } 
}


function endCall() {
    stopBeeping();
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    
    const ringBox = document.getElementById('ringBox');
    const callButton = document.getElementById('callButton');
    
    if (isConnected) {
        const summary = formatCallSummary(seconds);
        callStatus.textContent = summary;
        endCallBtn.style.display = 'none';
        
        // Show summary for 3 seconds then reset UI
        setTimeout(() => {
            ringBox.style.display = 'none';
            callButton.style.display = 'block';
            callStatus.textContent = 'Ready to call';
        }, 3000);
    }
    
    isConnected = false;
    socket = null;
}

callButton.addEventListener('click', startCall);
endCallBtn.addEventListener('click', endCall);

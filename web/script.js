const historyDiv = document.getElementById("history-div");
const numberInput = document.getElementById("number-input");
const newButton = document.getElementById("new-button");
const sizeInput = document.getElementById("size-input");
const submitButton = document.getElementById("submit-button");
const rollSpan = document.getElementById("roll");
const robotDiceSpan = document.getElementById("robot-dice");
const lieLink = document.getElementById("lie-link");
const callLeadSpan = document.getElementById("call-lead-span");

let Ds = [1, 1];
const SIDES = 6;
let D_PRI = Math.max(Ds[0], Ds[1]) * SIDES + 2;
let PRI_INDEX = D_PRI - 2;
let N_ACTIONS = (Ds[0] + Ds[1]) * SIDES + 1;
let CUR_INDEX = N_ACTIONS;
let D_PUB_PER_PLAYER = N_ACTIONS + 1;
let D_PUB = 2 * D_PUB_PER_PLAYER;

let rs = [[], []];
let last_call = -1;
let humanId = 0;
let privs = [null, null];
let state = null;

const phrases = [
   "I'll say",
   "Maybe",
   "What about",
   "I guess",
   "OK,",
   "Then I say",
   "I say",
   "Aha,",
   "Hmm,"
]

let session = {};

////////////////////////////////////////////////////////////////////////////////
// Model stuff
////////////////////////////////////////////////////////////////////////////////

const modelNames = {};
modelNames[[1,1]] = "./model_11_joker.onnx";

modelNames[[1,2]] = "./model_12_joker.onnx";
modelNames[[2,1]] = "./model_21_joker.onnx";
modelNames[[2,2]] = "./model_22_joker.onnx";

modelNames[[1,3]] = "./model_13_joker.onnx";
modelNames[[3,1]] = "./model_31_joker.onnx";
modelNames[[2,3]] = "./model_23_joker.onnx";
modelNames[[3,2]] = "./model_32_joker.onnx";
modelNames[[3,3]] = "./model_33_joker.onnx";

async function value(state, priv) {
   const res = await session[Ds].run({ priv: priv, pub: state });
   return res.value.data[0];
}

// Load our model.
async function main() {
   newButton.addEventListener("mousedown", newGameClicked);
   submitButton.addEventListener("mousedown", submit);
   lieLink.addEventListener("mousedown", submitLie);

   for (let i = 1; i <= SIDES; i++) {
      const myi = i;
      const elem = document.getElementById("dice-"+i);
      elem.addEventListener("mousedown", event => clickDice(elem, myi));
   }

   await newGame(3, 3, -1);
}

main();

async function newGame(D1, D2, newHumanId) {
   Ds = [D1, D2];
   D_PRI = Math.max(Ds[0], Ds[1]) * SIDES + 2;
   PRI_INDEX = D_PRI - 2;

   N_ACTIONS = (Ds[0] + Ds[1]) * SIDES + 1;
   CUR_INDEX = N_ACTIONS;
   D_PUB_PER_PLAYER = N_ACTIONS + 1;
   D_PUB = 2 * D_PUB_PER_PLAYER;

   if (!(Ds in session)) {
      console.log("Loading model " + modelNames[Ds]);
      session[Ds] = await ort.InferenceSession.create(modelNames[Ds]);
      console.log("Done.");
   }

   if (newHumanId === -1) {
      humanId = Math.floor(Math.random() * 2);
   } else {
      humanId = newHumanId;
   }
   console.log("Human id: " + humanId);

   last_call = -1;

   for (let p = 0; p < 2; p++) {
      rs[p].length = 0; // Clear previous roll
      privs[p] = new ort.Tensor(
         "float32",
         new Float32Array(Array(D_PRI).fill(0))
      );
      privs[p].data[PRI_INDEX + p] = 1;

      for (let i = 0; i < Ds[p]; i++) {
         const r = Math.floor(Math.random() * SIDES);
         rs[p].push(r + 1);
         privs[p].data[r] = 1;
      }
      rs[p].sort();
   }
   state = new ort.Tensor("float32", new Float32Array(Array(D_PUB).fill(0)));
   state.data[CUR_INDEX] = 1;

   function empty(element) {
      while (element.firstChild)
         element.removeChild(element.firstChild);
   }

   empty(historyDiv);
   empty(rollSpan);
   empty(robotDiceSpan);
   for (let i = 0; i < Ds[humanId]; i++) {
      rollSpan.appendChild(newDiceIcon(rs[humanId][i]));
      //rollSpan.appendChild(document.createTextNode(" "));
   }
   robotDiceSpan.appendChild(document.createTextNode(Ds[1-humanId].toString()));
   numberInput.setAttribute("max", Ds[0] + Ds[1]);

   lieLink.classList.add('hidden');
   empty(callLeadSpan);
   callLeadSpan.appendChild(document.createTextNode("Your call: "));

   if (humanId === 0)
      addStringToHistory("You start...");
   else
      addStringToHistory("🤖 Starts...");

   if (humanId !== 0) await goRobot();
}

function newGameClicked() {
   const n = Number.parseInt(sizeInput.value, 10);
   if (n === undefined || n > 2 || n < 1) {
      console.log("Unsupported size", sizeInput.value);
   }
   newGame(n, n, -1);
}

function newDiceIcon(i) {
   const elem = document.createElement("i");
   elem.className = "fas fa-dice-" + ['one', 'two', 'three', 'four', 'five', 'six'][i-1];
   elem.classList.add("small-dice");
   return elem;
}


function clickDice(elem, i) {
   for (let i = 1; i <= SIDES; i++) {
      const elem = document.getElementById("dice-"+i);
      elem.classList.remove('clicked');
   }
   elem.classList.add('clicked');
}



function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function addElementToHistory(elem, class_) {
   const para = document.createElement("div");
   para.classList.add("new-call");
   para.classList.add(class_);
   para.appendChild(elem);
   historyDiv.appendChild(para);
   await sleep(500);
}

async function addStringToHistory(string, class_) {
   const text = document.createTextNode(string);
   await addElementToHistory(text, class_);
}


function actionToSpan(prefix, action, postfix) {
   const span = document.createElement("span");
   if (action === N_ACTIONS - 1) {
      span.appendChild(document.createTextNode(prefix + "liar!" + postfix));
   } else {
      const n = Math.floor(action / SIDES) + 1;
      const d = (action % SIDES) + 1;

      span.appendChild(document.createTextNode(prefix + n + " times "));
      span.appendChild(newDiceIcon(d));
      span.appendChild(document.createTextNode(postfix));
   }
   return span
}

async function submitLie(event) {
   const action = N_ACTIONS - 1;
   _apply_action(state, action);
   const oldCall = last_call;
   last_call = action;
   lieLink.classList.add('hidden');
   await addElementToHistory(actionToSpan("", action, ""), 'human-call');
   endGame(oldCall, false);
}

async function submit(event) {
   let action;
   const n = Number.parseInt(numberInput.value, 10);
   if (n !== n || n < 1 || n > Ds[0]+Ds[1]) {
      console.log("Bad n: " + n);
      return;
   }

   let d = 0;
   for (let i = 1; i <= SIDES; i++) {
      const elem = document.getElementById("dice-"+i);
      if (elem.classList.contains('clicked'))
         d = i;
   }
   if (d == 0) {
      console.log("No die selected");
      return;
   }

   action = (n - 1) * SIDES + (d - 1);

   if (action <= last_call) {
      console.log("Call is too low");
      return;
   }

   _apply_action(state, action);
   const oldCall = last_call;
   last_call = action;

   lieLink.classList.add('hidden');
   await addElementToHistory(actionToSpan("", action, ""), 'human-call');
   await goRobot();
}

async function goRobot() {
   const action = await sampleCallFromPolicy();
   _apply_action(state, action);
   const oldCall = last_call;
   last_call = action;
   const prefix = phrases[Math.floor(Math.random() * phrases.length)];
   await addElementToHistory(actionToSpan("🤖: "+prefix+" " , action, ""), 'robot-call');
   if (action === N_ACTIONS - 1) {
      endGame(oldCall, true);
   }
   else {
      lieLink.classList.remove('hidden');
   }
}

function evaluate(call) {
   // Returns true if call was true (not the more recent "lie")
   if (call === -1) return true;

   const n = Math.floor(call / SIDES) + 1;
   const d = (call % SIDES) + 1;

   let actual = 0;
   for (let p = 0; p < 2; p++) {
      for (let i = 0; i < rs[p].length; i++) {
         if (rs[p][i] === d || rs[p][i] === 1) actual += 1;
      }
   }

   return actual >= n;
}

function endGame(call, isRoboCall) {
   const isGood = evaluate(call);
   addElementToHistory(actionToSpan("The call \"", call, "\" was " + isGood + "!"));

   const span = document.createElement("span");
   span.appendChild(document.createTextNode("The rolls were "));
   for (let i = 0; i < Ds[0]; i++) {
      span.appendChild(newDiceIcon(rs[0][i]));
   }
   span.appendChild(document.createTextNode(" and "));
   for (let i = 0; i < Ds[1]; i++) {
      span.appendChild(newDiceIcon(rs[1][i]));
   }
   addElementToHistory(span);

   let newDs = [...Ds];
   const robotWon = (!isRoboCall && isGood) || (isRoboCall && !isGood);
   if (robotWon) {
      newDs[1 - humanId] -= 1;
   } else {
      newDs[humanId] -= 1;
   }

   // If we continue for more rounds
   if (newDs[0] > 0 && newDs[1] > 0) {
      if (robotWon) {
         addStringToHistory("🤖 wins the round!");
      } else {
         addStringToHistory("🎉 You win the round!");
      }

      const continueLink = document.createElement("span");
      continueLink.classList.add("link");
      continueLink.appendChild(document.createTextNode("Continue..."));
      continueLink.addEventListener("mousedown", () => {
         // TODO: Should the loser start?
         newGame(newDs[0], newDs[1], humanId);
      });
      addElementToHistory(continueLink);
   }
   // Game over
   else {
      if (robotWon) {
         addStringToHistory("🤖 wins the game!");
      } else {
         addStringToHistory("🎉 You win the game!");
      }
   }
}

// Game functions

function get_cur(state) {
   return 1 - state.data[CUR_INDEX];
}

function _apply_action(state, action) {
   const cur = get_cur(state);
   state.data[action + cur * D_PUB_PER_PLAYER] = 1;
   state.data[CUR_INDEX + cur * D_PUB_PER_PLAYER] = 0;
   state.data[CUR_INDEX + (1 - cur) * D_PUB_PER_PLAYER] = 1;
}

function weightedChoice(array, dist) {
   let r = Math.random();
   return array.find((e, i) => (r -= dist[i]) < 0);
}

async function sampleCallFromPolicy() {

   console.log("State:", [... state.data]);
   console.log("Private:", [... privs[1-humanId].data]);

   const n_actions = N_ACTIONS - last_call - 1;
   const v = await value(state, privs[1 - humanId]);

   const actions = [];
   const regrets = [];
   let sum = 0;
   for (let i = 0; i < n_actions; i++) {
      const clonedState = new ort.Tensor("float32", new Float32Array([...state.data]));
      const action = i + last_call + 1;
      actions.push(action);
      _apply_action(clonedState, i + last_call + 1);
      const vi = await value(clonedState, privs[1 - humanId]);
      const r = Math.max(vi - v, 0);
      regrets.push(r);
      sum += r;
   }

   if (sum === 0) {
      for (let i = 0; i < n_actions; i++) {
         regrets[i] = 1 / n_actions;
      }
   } else {
      for (let i = 0; i < n_actions; i++) {
         regrets[i] /= sum;
      }
   }

   console.log("Probabilities:", regrets);

   return weightedChoice(actions, regrets);
}
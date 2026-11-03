const defaultChatSettings = {
  model: "gpt-4o-mini",
  embedding_model: "text-embedding-3-small",
  top_k: 4,
  memory_mode: "buffer",
  budget_tokens: 700,
  show_reasoning: false,
  show_interaction_cost: false,
};
const defaultChatModels = ["gpt-4o-mini", "gpt-4.1-mini"];
const defaultEmbeddingModels = ["text-embedding-3-small", "text-embedding-3-large"];

const state = {
  view: "chat",
  records: [],
  medicines: [],
  editing: null,
  searchTimer: null,
  toastTimer: null,
  sessionId: localStorage.getItem("ifarmaco-chat-session") || crypto.randomUUID(),
  chatHistory: [],
  chatBusy: false,
  chatConfigured: false,
  chatSettings: { ...defaultChatSettings },
  settingsDirty: false,
};

const viewConfig = {
  medicines: {
    title: "Farmaci",
    description: "Consulta e aggiorna il catalogo dei farmaci salvato nel database SQLite.",
    endpoint: "/api/medicines",
    recordsTitle: "Elenco farmaci",
    recordsDescription: "Dati importati dal catalogo AIFA e modificabili localmente.",
    addLabel: "Aggiungi farmaco",
    countLabel: (count) => `${count} farmaci`,
  },
  stock: {
    title: "Scorte",
    description: "Gestisci lotti, date di scadenza e quantità disponibili per ogni farmaco.",
    endpoint: "/api/stock",
    recordsTitle: "Lotti di scorta",
    recordsDescription: "Ogni riga rappresenta un lotto con scadenza e quantità proprie.",
    addLabel: "Aggiungi lotto",
    countLabel: (count) => `${count} lotti`,
  },
};

const medicineFields = [
  { name: "id", label: "ID farmaco", required: true, placeholder: "Es. 0000012345" },
  { name: "nome", label: "Nome", required: true, placeholder: "Nome del farmaco" },
  { name: "casa_produttrice", label: "Casa produttrice", placeholder: "Azienda" },
  { name: "formato", label: "Formato", placeholder: "Es. Compressa" },
  { name: "quantita", label: "Dosaggio", placeholder: "Es. 0,2 g" },
  { name: "bugiardino_url", label: "Link al bugiardino", type: "url", placeholder: "https://..." },
  { name: "bugiardino", label: "Bugiardino PDF", type: "file", accept: ".pdf,application/pdf", required: true },
];

const stockFields = [
  { name: "farmaco_id", label: "Farmaco", type: "medicine", required: true },
  { name: "data_scadenza", label: "Data di scadenza", type: "date", required: true },
  { name: "nr_pezzi", label: "Numero di pezzi", type: "number", min: "0", step: "1", required: true },
];

const elements = {
  nav: [...document.querySelectorAll("[data-view]")],
  chatView: document.querySelector("#chat-view"),
  settingsView: document.querySelector("#settings-view"),
  databaseView: document.querySelector("#database-view"),
  breadcrumb: document.querySelector("#breadcrumb-current"),
  databaseTitle: document.querySelector("#database-title"),
  databaseDescription: document.querySelector("#database-description"),
  recordsTitle: document.querySelector("#records-title"),
  recordsDescription: document.querySelector("#records-description"),
  addButton: document.querySelector("#add-record"),
  addLabel: document.querySelector("#add-record-label"),
  search: document.querySelector("#record-search"),
  count: document.querySelector("#record-count"),
  tableHead: document.querySelector("#records-head"),
  tableBody: document.querySelector("#records-body"),
  emptyState: document.querySelector("#empty-state"),
  dialog: document.querySelector("#record-dialog"),
  form: document.querySelector("#record-form"),
  dialogTitle: document.querySelector("#dialog-title"),
  dialogFields: document.querySelector("#dialog-fields"),
  closeDialog: document.querySelector("#close-dialog"),
  cancelDialog: document.querySelector("#cancel-dialog"),
  toast: document.querySelector("#toast-message"),
  chatForm: document.querySelector("#chat-form"),
  question: document.querySelector("#question"),
  conversation: document.querySelector("#conversation"),
  answerAvatar: document.querySelector(".answer-avatar"),
  chatModel: document.querySelector("#chat-model"),
  chatEmbeddingModel: document.querySelector("#chat-embedding-model"),
  chatTopK: document.querySelector("#chat-top-k"),
  chatMemory: document.querySelector("#chat-memory"),
  chatBudget: document.querySelector("#chat-budget"),
  chatShowReasoning: document.querySelector("#chat-show-reasoning"),
  chatShowInteractionCost: document.querySelector("#chat-show-interaction-cost"),
  settingsForm: document.querySelector("#settings-form"),
  saveSettings: document.querySelector("#save-settings"),
  chatStatus: document.querySelector("#chat-status"),
  chatStatusDot: document.querySelector("#chat-status-dot"),
  chatSetupNotice: document.querySelector("#chat-setup-notice"),
  newChat: document.querySelector("#new-chat"),
  sendQuestion: document.querySelector("#send-question"),
};

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...options.headers,
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Richiesta non riuscita (${response.status}).`);
  }
  return payload;
}

function showToast(message, isError = false) {
  clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.classList.add("is-visible");
  state.toastTimer = setTimeout(() => elements.toast.classList.remove("is-visible"), 3500);
}

function showView(view) {
  state.view = view;
  const isChat = view === "chat";
  const isSettings = view === "settings";
  elements.chatView.hidden = !isChat;
  elements.settingsView.hidden = !isSettings;
  elements.databaseView.hidden = isChat || isSettings;
  elements.nav.forEach((link) => {
    const active = link.dataset.view === view;
    link.classList.toggle("active", active);
    if (active) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  if (isChat) {
    elements.breadcrumb.textContent = "Nuova domanda";
    return;
  }

  if (isSettings) {
    elements.breadcrumb.textContent = "Impostazioni";
    return;
  }

  const config = viewConfig[view];
  elements.breadcrumb.textContent = config.title;
  elements.databaseTitle.textContent = config.title;
  elements.databaseDescription.textContent = config.description;
  elements.recordsTitle.textContent = config.recordsTitle;
  elements.recordsDescription.textContent = config.recordsDescription;
  elements.addLabel.textContent = config.addLabel;
  elements.search.placeholder = view === "medicines"
    ? "Cerca per nome, codice o produttore..."
    : "Cerca per farmaco, codice o scadenza...";
  loadDatabaseView();
}

async function loadSummary() {
  const summary = await requestJSON("/api/summary");
  document.querySelector("#summary-medicines").textContent = summary.farmaci.toLocaleString("it-IT");
  document.querySelector("#summary-lots").textContent = summary.lotti.toLocaleString("it-IT");
  document.querySelector("#summary-pieces").textContent = summary.pezzi.toLocaleString("it-IT");
}

async function loadMedicines() {
  state.medicines = await requestJSON("/api/medicines");
}

async function loadDatabaseView() {
  elements.tableBody.replaceChildren();
  elements.count.textContent = "Caricamento...";
  elements.emptyState.hidden = true;
  try {
    await loadSummary();
    if (state.view === "stock") {
      await loadMedicines();
    }
    const config = viewConfig[state.view];
    const query = elements.search.value.trim();
    const suffix = query ? `?q=${encodeURIComponent(query)}` : "";
    state.records = await requestJSON(`${config.endpoint}${suffix}`);
    renderTable();
  } catch (error) {
    elements.count.textContent = "";
    showToast(error.message, true);
  }
}

function appendHeaderRow(labels) {
  const row = document.createElement("tr");
  labels.forEach((label) => {
    const header = document.createElement("th");
    header.scope = "col";
    header.textContent = label;
    row.append(header);
  });
  elements.tableHead.replaceChildren(row);
}

function createCell(content, className = "") {
  const cell = document.createElement("td");
  if (className) {
    cell.className = className;
  }
  if (content instanceof Node) {
    cell.append(content);
  } else {
    cell.textContent = content ?? "—";
  }
  return cell;
}

function createEditButton(recordId) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "table-action";
  button.dataset.action = "edit";
  button.dataset.id = String(recordId);
  button.setAttribute("aria-label", "Modifica record");
  button.textContent = "Modifica";
  return button;
}

function renderMedicines() {
  appendHeaderRow(["ID", "Farmaco", "Produttore", "Formato e dosaggio", "Scorte", ""]);
  const rows = state.records.map((medicine) => {
    const row = document.createElement("tr");
    row.append(createCell(medicine.id, "code-cell"));

    const name = document.createElement("div");
    name.className = "medicine-name";
    name.textContent = medicine.nome;
    const nameCell = document.createElement("td");
    nameCell.append(name);
    row.append(nameCell);

    row.append(createCell(medicine.casa_produttrice));
    const form = document.createElement("div");
    form.className = "medicine-form";
    form.textContent = medicine.formato || "Formato non indicato";
    const dosage = document.createElement("small");
    dosage.textContent = medicine.quantita || "Dosaggio non indicato";
    const formCell = document.createElement("td");
    formCell.append(form, dosage);
    row.append(formCell);

    const stockInfo = document.createElement("div");
    stockInfo.className = "stock-cell";
    stockInfo.textContent = `${medicine.pezzi_totali} pezzi`;
    const lots = document.createElement("small");
    lots.textContent = `${medicine.numero_lotti} lotti`;
    const stockCell = document.createElement("td");
    stockCell.append(stockInfo, lots);
    row.append(stockCell);

    const actions = document.createElement("td");
    actions.append(createEditButton(medicine.id));
    row.append(actions);
    return row;
  });
  elements.tableBody.replaceChildren(...rows);
}

function renderStock() {
  appendHeaderRow(["Farmaco", "ID farmaco", "Scadenza", "Disponibilità", ""]);
  const rows = state.records.map((lot) => {
    const row = document.createElement("tr");
    const name = document.createElement("div");
    name.className = "medicine-name";
    name.textContent = lot.farmaco_nome;
    const lotId = document.createElement("small");
    lotId.textContent = `Lotto #${lot.id}`;
    const medicineCell = document.createElement("td");
    medicineCell.append(name, lotId);
    row.append(medicineCell, createCell(lot.farmaco_id, "code-cell"));

    const expiry = document.createElement("span");
    const isExpired = lot.data_scadenza < new Date().toISOString().slice(0, 10);
    expiry.className = `expiry-badge ${isExpired ? "expired" : "valid"}`;
    expiry.textContent = new Date(`${lot.data_scadenza}T00:00:00`).toLocaleDateString("it-IT");
    row.append(createCell(expiry));

    const quantity = document.createElement("span");
    quantity.className = "quantity-value";
    quantity.textContent = String(lot.nr_pezzi);
    const pieces = document.createElement("small");
    pieces.textContent = "pezzi";
    const quantityCell = document.createElement("td");
    quantityCell.append(quantity, pieces);
    row.append(quantityCell);

    const actions = document.createElement("td");
    actions.className = "action-group";
    actions.append(createEditButton(lot.id));
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "table-action delete-action";
    removeButton.dataset.action = "delete";
    removeButton.dataset.id = String(lot.id);
    removeButton.textContent = "Elimina";
    actions.append(removeButton);
    row.append(actions);
    return row;
  });
  elements.tableBody.replaceChildren(...rows);
}

function renderTable() {
  if (state.view === "medicines") {
    renderMedicines();
  } else {
    renderStock();
  }
  elements.count.textContent = viewConfig[state.view].countLabel(state.records.length);
  elements.emptyState.hidden = state.records.length > 0;
}

function addField(field, value = "") {
  const label = document.createElement("label");
  label.className = "form-field";
  const caption = document.createElement("span");
  caption.textContent = field.label;
  label.append(caption);

  let input;
  if (field.type === "medicine") {
    input = document.createElement("select");
    input.name = field.name;
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Seleziona un farmaco";
    input.append(placeholder);
    state.medicines.forEach((medicine) => {
      const option = document.createElement("option");
      option.value = medicine.id;
      option.textContent = `${medicine.nome} · ${medicine.id}`;
      input.append(option);
    });
  } else {
    input = document.createElement("input");
    input.name = field.name;
    input.type = field.type || "text";
    if (field.type === "file" && state.editing) {
      return;
    }
    if (field.accept) {
      input.accept = field.accept;
    }
    if (field.placeholder) {
      input.placeholder = field.placeholder;
    }
    if (field.min !== undefined) {
      input.min = field.min;
    }
    if (field.step !== undefined) {
      input.step = field.step;
    }
    if (field.name === "id" && state.editing) {
      input.readOnly = true;
    }
  }

  input.value = value ?? "";
  input.required = Boolean(field.required && !(field.type === "file" && state.editing));
  label.append(input);
  elements.dialogFields.append(label);
}

async function openEditor(record = null) {
  state.editing = record;
  elements.dialogFields.replaceChildren();
  const isMedicine = state.view === "medicines";
  const fields = isMedicine ? medicineFields : stockFields;
  if (!isMedicine) {
    try {
      await loadMedicines();
    } catch (error) {
      showToast(error.message, true);
      return;
    }
  }
  elements.dialogTitle.textContent = record
    ? (isMedicine ? "Modifica farmaco" : "Modifica lotto")
    : (isMedicine ? "Nuovo farmaco" : "Nuovo lotto");
  fields.forEach((field) => addField(field, record?.[field.name]));
  elements.dialog.showModal();
  elements.dialogFields.querySelector("input, select")?.focus();
}

async function saveRecord(event) {
  event.preventDefault();
  const formData = new FormData(elements.form);
  const isMedicine = state.view === "medicines";
  const isNewMedicine = isMedicine && !state.editing;
  const payload = Object.fromEntries(formData.entries());
  delete payload.bugiardino;
  if (state.view === "stock") {
    payload.nr_pezzi = Number(payload.nr_pezzi);
  }
  const base = isMedicine ? "/api/medicines" : "/api/stock";
  const id = state.editing?.id;
  const url = state.editing ? `${base}/${encodeURIComponent(id)}` : base;
  const method = state.editing ? "PUT" : "POST";
  try {
    const result = await requestJSON(url, {
      method,
      body: isNewMedicine ? formData : JSON.stringify(payload),
    });
    elements.dialog.close();
    if (isNewMedicine && result.embedding_status !== "indexed") {
      showToast(result.embedding_message || "Farmaco salvato, ma non indicizzato.", true);
    } else {
      showToast(state.editing ? "Modifiche salvate." : "Record aggiunto.");
    }
    await loadDatabaseView();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function deleteStock(stockId) {
  if (!window.confirm("Vuoi eliminare questo lotto di scorta?")) {
    return;
  }
  try {
    await requestJSON(`/api/stock/${encodeURIComponent(stockId)}`, { method: "DELETE" });
    showToast("Lotto eliminato.");
    await loadDatabaseView();
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderChat(messages = [], pending = false) {
  elements.conversation.replaceChildren();
  if (!messages.length && !pending) {
    const welcome = document.createElement("p");
    welcome.className = "chat-welcome";
    welcome.textContent = "Chiedimi informazioni su uno o più farmaci: puoi citarli nella domanda.";
    elements.conversation.append(welcome);
  }
  messages.forEach((message) => {
    if (message.role === "user") {
      const row = document.createElement("div");
      row.className = "message user-message";
      const label = document.createElement("span");
      label.className = "message-label";
      label.textContent = message.content.startsWith("Riassunto della conversazione:")
        ? "MEMORIA"
        : "LA TUA DOMANDA";
      const content = document.createElement("p");
      content.textContent = message.content;
      row.append(label, content);
      elements.conversation.append(row);
      return;
    }

    const row = document.createElement("div");
    row.className = "answer-row";
    const avatar = elements.answerAvatar.cloneNode(true);
    const content = document.createElement("div");
    content.className = "answer-content";
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = "RISPOSTA";
    const answer = document.createElement("p");
    answer.textContent = message.content;
    content.append(label, answer);
    (message.sources || []).forEach((source) => {
      const link = document.createElement("a");
      link.className = "source-card";
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.append(document.createTextNode("Bugiardino "));
      const page = document.createElement("strong");
      page.textContent = `pagina ${source.page}`;
      link.append(page);
      content.append(link);
    });
    const showReasoning = state.chatSettings.show_reasoning && message.reasoning;
    const showCost = state.chatSettings.show_interaction_cost
      && Number.isFinite(message.interaction_cost_usd);
    if (showReasoning || showCost) {
      const details = document.createElement("div");
      details.className = "answer-details";
      if (showReasoning) {
        const explanation = document.createElement("p");
        explanation.className = "answer-reasoning";
        const label = document.createElement("strong");
        label.textContent = "Sintesi";
        explanation.append(label, document.createTextNode(message.reasoning));
        details.append(explanation);
      }
      if (showCost) {
        const cost = document.createElement("p");
        cost.className = "answer-cost";
        cost.textContent = `Costo stimato: ${new Intl.NumberFormat("it-IT", {
          style: "currency",
          currency: "USD",
          minimumFractionDigits: 6,
          maximumFractionDigits: 8,
        }).format(message.interaction_cost_usd)} · calcolo indicativo sui token API`;
        details.append(cost);
      }
      content.append(details);
    }
    row.append(avatar, content);
    elements.conversation.append(row);
  });
  if (pending) {
    const row = document.createElement("div");
    row.className = "answer-row is-pending";
    row.setAttribute("role", "status");
    const avatar = elements.answerAvatar.cloneNode(true);
    const content = document.createElement("div");
    content.className = "answer-content";
    const text = document.createElement("p");
    text.textContent = "Sto consultando il bugiardino...";
    content.append(text);
    row.append(avatar, content);
    elements.conversation.append(row);
  }
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

async function initializeChat() {
  localStorage.setItem("ifarmaco-chat-session", state.sessionId);
  try {
    const [config, settings] = await Promise.all([
      requestJSON("/api/chat/config"),
      requestJSON("/api/chat/settings"),
    ]);
    elements.chatModel.replaceChildren();
    (config.models || defaultChatModels).forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      elements.chatModel.append(option);
    });
    elements.chatEmbeddingModel.replaceChildren();
    (config.embedding_models || defaultEmbeddingModels).forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      elements.chatEmbeddingModel.append(option);
    });
    state.chatSettings = { ...defaultChatSettings, ...settings };
    elements.chatModel.value = state.chatSettings.model;
    elements.chatEmbeddingModel.value = state.chatSettings.embedding_model;
    elements.chatTopK.value = state.chatSettings.top_k;
    elements.chatMemory.value = state.chatSettings.memory_mode;
    elements.chatBudget.value = state.chatSettings.budget_tokens;
    elements.chatShowReasoning.checked = state.chatSettings.show_reasoning;
    elements.chatShowInteractionCost.checked = state.chatSettings.show_interaction_cost;
    elements.chatStatus.textContent = config.configured ? "Pronto ad aiutarti" : "API non configurata";
    state.chatConfigured = config.configured;
    elements.chatStatusDot.classList.toggle("is-offline", !config.configured);
    elements.sendQuestion.disabled = !config.configured;
    elements.chatSetupNotice.hidden = config.configured;
    if (!config.configured) {
      elements.chatSetupNotice.textContent = "Aggiungi OPENAI_API_KEY al file .env e riavvia l'applicazione per attivare la chat.";
    }
    const saved = await requestJSON(`/api/chat/session/${encodeURIComponent(state.sessionId)}`);
    state.chatHistory = saved.messages;
    renderChat(state.chatHistory);
  } catch (error) {
    elements.chatStatus.textContent = "Servizio non disponibile";
    elements.chatStatusDot.classList.add("is-offline");
    showToast(error.message, true);
  }
}

async function sendChat(event) {
  event.preventDefault();
  if (state.settingsDirty) {
    showToast("Salva le impostazioni prima di inviare una domanda.", true);
    return;
  }
  if (!state.chatConfigured) {
    showToast("La chat non è configurata. Riavvia l'app dopo aver aggiornato .env.", true);
    return;
  }
  const message = elements.question.value.trim();
  if (!message || state.chatBusy) {
    return;
  }
  state.chatBusy = true;
  elements.sendQuestion.disabled = true;
  elements.question.value = "";
  elements.chatStatus.textContent = "Sto cercando nel bugiardino";
  renderChat([...state.chatHistory, { role: "user", content: message }], true);
  try {
    const result = await requestJSON("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        message,
        ...state.chatSettings,
      }),
    });
    state.chatHistory = result.messages;
    renderChat(state.chatHistory);
    elements.chatStatus.textContent = result.searched_query !== message
      ? `Ricerca: ${result.searched_query}`
      : "Pronto ad aiutarti";
  } catch (error) {
    renderChat(state.chatHistory);
    if (!elements.question.value.trim()) {
      elements.question.value = message;
    }
    showToast(error.message, true);
    elements.chatStatus.textContent = "Errore nella risposta";
  } finally {
    state.chatBusy = false;
    elements.sendQuestion.disabled = !state.chatConfigured;
    elements.question.focus();
  }
}

async function saveChatSettings(event) {
  event.preventDefault();
  const settings = {
    model: elements.chatModel.value,
    embedding_model: elements.chatEmbeddingModel.value,
    top_k: Number(elements.chatTopK.value),
    memory_mode: elements.chatMemory.value,
    budget_tokens: Number(elements.chatBudget.value),
    show_reasoning: elements.chatShowReasoning.checked,
    show_interaction_cost: elements.chatShowInteractionCost.checked,
  };
  elements.saveSettings.disabled = true;
  try {
    state.chatSettings = await requestJSON("/api/chat/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
    state.settingsDirty = false;
    renderChat(state.chatHistory);
    showToast("Impostazioni salvate.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.saveSettings.disabled = false;
  }
}

function startNewChat() {
  state.sessionId = crypto.randomUUID();
  state.chatHistory = [];
  localStorage.setItem("ifarmaco-chat-session", state.sessionId);
  elements.chatStatus.textContent = "Pronto ad aiutarti";
  renderChat();
}

elements.nav.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    showView(link.dataset.view);
  });
});

elements.addButton.addEventListener("click", () => openEditor());
elements.form.addEventListener("submit", saveRecord);
elements.closeDialog.addEventListener("click", () => elements.dialog.close());
elements.cancelDialog.addEventListener("click", () => elements.dialog.close());

elements.dialog.addEventListener("click", (event) => {
  if (event.target === elements.dialog) {
    elements.dialog.close();
  }
});

elements.search.addEventListener("input", () => {
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(loadDatabaseView, 250);
});

elements.tableBody.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }
  const record = state.records.find((item) => String(item.id) === button.dataset.id);
  if (button.dataset.action === "edit" && record) {
    openEditor(record);
  } else if (button.dataset.action === "delete") {
    deleteStock(button.dataset.id);
  }
});

elements.chatForm.addEventListener("submit", sendChat);
elements.settingsForm.addEventListener("submit", saveChatSettings);
elements.settingsForm.addEventListener("input", () => {
  state.settingsDirty = true;
});
elements.settingsForm.addEventListener("change", () => {
  state.settingsDirty = true;
});
elements.newChat.addEventListener("click", startNewChat);
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});
document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    elements.question.value = button.textContent.trim();
    elements.question.focus();
  });
});

initializeChat();

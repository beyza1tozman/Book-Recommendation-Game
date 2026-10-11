const LOCAL_API_URL = "http://127.0.0.1:8000";
const DEPLOYED_API_URL = "https://book-recommendation-game.onrender.com"; // your Render service URL

const isLocal = ["localhost", "127.0.0.1", ""].includes(window.location.hostname);
const API_BASE_URL = isLocal ? LOCAL_API_URL : DEPLOYED_API_URL;

const bookshopOwner = document.getElementById("bookshopowner");
const dialogueArea = document.getElementById("dialogueArea");
const nextDialogueBtn = document.getElementById("nextDialogue");
const modal = document.getElementById("recommendModal");
const closeDialogue = document.getElementById("closeDialogue");
const closeModal = document.getElementById("closeModal");
const bookInput = document.getElementById("bookInput");
const resultsDiv = document.getElementById("results");
const submitBtn = document.getElementById("submitBook");
const template = document.getElementById("bookCardTemplate");

bookshopOwner.addEventListener("click", () => {
    dialogueArea.style.display = "block";
});

nextDialogueBtn.addEventListener("click", () => {
    dialogueArea.style.display = "none";
    modal.style.display = "flex";
});

closeDialogue.addEventListener("click", () => {
    dialogueArea.style.display = "none";
});

closeModal.addEventListener("click", () => {
    modal.style.display = "none";
    bookInput.value = "";
    resultsDiv.innerHTML = "";
});

bookInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        submitBtn.click();
    }
});

submitBtn.addEventListener("click", async () => {
    const title = bookInput.value.trim();

    if (!title) {
        resultsDiv.innerHTML = `<p class="error-message">Enter a book name!</p>`;
        return;
    }

    resultsDiv.innerHTML = "Loading...";

    // the free server sleeps when idle, so the first request can take a while
    const wakeUpNotice = setTimeout(() => {
        resultsDiv.innerHTML = "Waking up the bookshop owner... this can take up to a minute.";
    }, 5000);

    try {
        const response = await fetch(`${API_BASE_URL}/recommend_book`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title })
        });
        clearTimeout(wakeUpNotice);

        if (!response.ok) {
            const err = await response.json();
            resultsDiv.innerHTML = `<p class="error-message">${err.detail}</p>`;
            return;
        }

        const data = await response.json();
        resultsDiv.innerHTML = "";

        data.forEach(book => {
            const card = template.content.cloneNode(true);

            card.querySelector("img").src = book.thumbnail;
            card.querySelector("img").alt = book.title;
            card.querySelector("strong").textContent = book.title;
            card.querySelector("em").textContent = book.authors;

            resultsDiv.appendChild(card);
        });

    } catch (err) {
        clearTimeout(wakeUpNotice);
        resultsDiv.innerHTML = `<p class="error-message">Couldn't reach the bookshop. Please try again in a moment.</p>`;
    }
});

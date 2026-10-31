const graphButtons = document.querySelectorAll('[data-image]');
const graphModal = document.getElementById('graphModal');
const graphImage = document.getElementById('graphImage');
const graphTitle = document.getElementById('graphTitle');
const graphDownload = document.getElementById('graphDownload');

const fileButtons = document.querySelectorAll('[data-file]');
const contentModal = document.getElementById('contentModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const modalDownload = document.getElementById('modalDownload');

function closeModal(modal) {
  modal.classList.remove('show');
}

function openGraphModal(imageSrc, title) {
  graphImage.src = imageSrc;
  graphTitle.textContent = title;
  graphDownload.href = imageSrc;
  graphDownload.setAttribute('download', title.replace(/\s+/g, '_') + '.png');
  graphImage.style.transform = 'scale(1)';
  graphImage.dataset.scale = '1';
  graphModal.classList.add('show');
}

function parseCSV(text) {
  const rows = text.trim().split(/\r?\n/);
  if (!rows.length) return [];

  const headers = rows[0].split(',');
  const values = rows.slice(1).map((row) => row.split(','));

  return values.map((row) => {
    const obj = {};
    headers.forEach((header, index) => {
      obj[header] = row[index] || '';
    });
    return obj;
  });
}

async function openDataModal(filePath, label) {
  const response = await fetch(filePath);
  const text = await response.text();
  const rows = parseCSV(text);

  modalTitle.textContent = label;
  modalDownload.href = filePath;
  modalDownload.setAttribute('download', label);

  if (!rows.length) {
    modalBody.innerHTML = '<p>No data available.</p>';
    contentModal.classList.add('show');
    return;
  }

  const headers = Object.keys(rows[0]);
  const table = document.createElement('table');
  table.className = 'data-table';

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  headers.forEach((header) => {
    const th = document.createElement('th');
    th.textContent = header;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement('tbody');
  rows.slice(0, 20).forEach((row) => {
    const tr = document.createElement('tr');
    headers.forEach((header) => {
      const td = document.createElement('td');
      td.textContent = row[header];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  modalBody.innerHTML = '';
  modalBody.appendChild(table);
  contentModal.classList.add('show');
}

graphButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const imageSrc = button.dataset.image;
    const title = button.dataset.title;
    openGraphModal(imageSrc, title);
  });
});

const graphPreviews = document.querySelectorAll('.graph-preview');
graphPreviews.forEach((img) => {
  img.addEventListener('click', () => {
    openGraphModal(img.dataset.image, img.dataset.title);
  });
});

fileButtons.forEach((button) => {
  button.addEventListener('click', async () => {
    const file = button.dataset.file;
    const label = button.dataset.label || button.textContent;
    await openDataModal(file, label);
  });
});

document.querySelectorAll('.close-btn').forEach((button) => {
  button.addEventListener('click', () => {
    const modal = button.closest('.modal');
    closeModal(modal);
  });
});

document.querySelectorAll('.modal').forEach((modal) => {
  modal.addEventListener('click', (event) => {
    if (event.target === modal) {
      closeModal(modal);
    }
  });
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeModal(graphModal);
    closeModal(contentModal);
  }
});

const zoomButtons = document.querySelectorAll('[data-zoom]');
zoomButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const current = Number(graphImage.dataset.scale || 1);
    const direction = Number(button.dataset.zoom);
    const next = Math.max(1, Math.min(3, current + direction * 0.5));
    graphImage.style.transform = `scale(${next})`;
    graphImage.dataset.scale = String(next);
  });
});

# UniRecommender: AI-Powered University Matching Engine

**UniRecommender** is a modern, responsive, and intelligent web application designed to help prospective students find their ideal university. By leveraging a sophisticated matching algorithm and a sleek, intuitive user interface, it provides personalized university recommendations based on academic profiles, financial budgets, and personal preferences.

![image](https://github.com/user-attachments/assets/eac09c3a-225e-4b62-932d-147e577bd952)
![image](https://github.com/user-attachments/assets/6392802f-54ec-452b-8f24-395c6553e3e0)

---

## ✨ Features

*   **Dual Application Types**: Tailored forms for both **Undergraduate** and **Graduate** applicants.
*   **Dynamic Student Profile**: Input GPA/CGPA, test scores (SAT, GRE, etc.), academic achievements, and more.
*   **Advanced Preference Filtering**: Filter by preferred countries, budget, field of study, and desired timeline.
*   **Weighted Priority System**: Customize the importance of factors like academic reputation, cost, location, and culture.
*   **Instant Recommendations**: Receive a ranked list of matching universities in a clean, two-column layout.
*   **Detailed University Cards**: Each recommendation includes:
    *   An overall match score, visualized with a custom gradient progress ring.
    *   Key details: global ranking, tuition fees, and acceptance rate.
    *   An expandable "Why this match?" section with a detailed breakdown of the scoring.
*   **Responsive Design**: A beautiful and intuitive UI that works seamlessly on both desktop and mobile devices.
*   **Modern Tech Stack**: Built with React, FastAPI, and Tailwind CSS for a high-performance, scalable solution.

---

## 🚀 Technology Stack

| Area      | Technology                                                                                                  |
| :-------- | :---------------------------------------------------------------------------------------------------------- |
| **Frontend** | [**React**](https://reactjs.org/) (with Hooks), [**Vite**](https://vitejs.dev/), [**Tailwind CSS**](https://tailwindcss.com/) |
| **Backend**  | [**Python 3**](https://www.python.org/), [**FastAPI**](https://fastapi.tiangolo.com/), [**Pydantic**](https://pydantic-docs.helpmanual.io/) |
| **Styling**  | [**Framer Motion**](https://www.framer.com/motion/) (for animations), [**Lucide React**](https://lucide.dev/) (icons) |
| **API Client** | [**Axios**](https://axios-http.com/)                                                                  |

---

## 🛠️ Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   **Node.js** (v18.x or later)
*   **npm** (v9.x or later)
*   **Python** (v3.9 or later)
*   `pip` and `venv`

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/UniRecommender.git
    cd UniRecommender
    ```

2.  **Setup the Backend (Python/FastAPI):**
    *   Navigate to the project root directory.
    *   Create and activate a virtual environment:
        ```bash
        # For Windows
        python -m venv venv
        .\venv\Scripts\activate

        # For macOS/Linux
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Setup the Frontend (React/Vite):**
    *   Navigate to the `frontend` directory:
        ```bash
        cd frontend
        ```
    *   Install the required npm packages:
        ```bash
        npm install
        ```

### Running the Application

You will need to run two separate processes in two different terminals.

1.  **Start the Backend Server:**
    *   From the **project root directory** (where `main.py` is located):
        ```bash
        uvicorn main:app --reload
        ```
    *   The API server will be running at `http://127.0.0.1:8000`.

2.  **Start the Frontend Development Server:**
    *   From the **`frontend` directory**:
        ```bash
        npm run dev
        ```
    *   The React application will be available at `http://localhost:5173` (or the next available port).

Open `http://localhost:5173` in your browser to see the application in action!

---

## 📁 Project Structure

```
.
├── frontend/         # React Frontend
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.jsx
│   └── ...
├── src/              # Python Backend
│   ├── api.py
│   ├── matching_algorithm.py
│   └── models.py
├── data/             # CSV and JSON data files
│   └── universities.csv
├── tests/            # Backend tests
├── main.py           # FastAPI app entry point
└── requirements.txt  # Python dependencies
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/UniRecommender/issues).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details. 

import "./App.css";
import BoardPage from "./components/BoardPage";
import SetupPage from "./components/SetupPage";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";


const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to="/setup" />,
  },
  {
    path: "/setup",
    element: <SetupPage />
  },
  {
    path: "/board",
    element: <BoardPage />
  }
])

function App() {
  return (
    <div className="App">
      <RouterProvider router={router} />
    </div>
  );
}

export default App;
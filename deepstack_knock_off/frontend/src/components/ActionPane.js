import { useState, useEffect } from "react";
import Action from "./Action";
import axios from 'axios';


const ActionPane = ({ gameState }) => {
  const [actions, setActions] = useState([]); // State to store actions from API
  const [selectedAction, setSelectedAction] = useState(null);

  const fetchLegalActions = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/legal-actions");
      setActions(response.data);
    } catch (error) {
      console.error('Failed to fetch legal actions:', error);
    }
  };

  useEffect(() => {
    fetchLegalActions();
  }, []);

  const handleActionClick = (action) => {
    setSelectedAction(action);
  };

  const applySelectedAction = () => {
    // Logic to apply the selected action
    console.log("Applying action:", selectedAction);
    // Reset selection after applying
    setSelectedAction(null);
  };

  const buttonClass = selectedAction ? "poker-button" : "poker-button disabled";

  return (
    <div className="action-pane">
      <div className="scroll-box">
        <ul>
          {actions.map((action, index) => (
            <Action
              key={index}
              action={action}
              onClick={() => handleActionClick(action)}
              isSelected={selectedAction === action}
            />
          ))}
        </ul>
      </div>
      <button className={buttonClass} onClick={applySelectedAction} disabled={!selectedAction}>Apply</button>
    </div>
  );
};

export default ActionPane;
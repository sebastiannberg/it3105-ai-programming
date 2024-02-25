import { useState } from "react";
import Action from "./Action";


const ActionPane = ({ gameState }) => {

  const actions = gameState.actions || ["Action 1", "Action 2", "Action 3", "Action 4"];

  const [selectedAction, setSelectedAction] = useState(null);

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
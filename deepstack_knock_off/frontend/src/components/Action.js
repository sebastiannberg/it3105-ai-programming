

const Action = ({ action, onClick, isSelected }) => {

  const className = `action-item ${isSelected ? 'selected' : ''}`;

  return (
    <li className={className} onClick={onClick}>
      <div>{action.action_name}</div>
    </li>
  );
};

export default Action;
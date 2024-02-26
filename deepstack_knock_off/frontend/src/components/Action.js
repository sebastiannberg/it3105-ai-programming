

const Action = ({ action, onClick, isSelected }) => {

  const className = `action-item ${isSelected ? 'selected' : ''}`;

  return (
    <li className={className} onClick={onClick}>
      <div>{action.name}</div>
    </li>
  );
};

export default Action;
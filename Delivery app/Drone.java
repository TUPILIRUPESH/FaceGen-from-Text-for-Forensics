class Drone extends Vehicle {
    @Override
    public void assign() {
        System.out.println("the delivery has been assigned in drone");
    }

    @Override
    public void dispatch() {
        System.out.println("the delivery has been dispatched successfully");
    }
}